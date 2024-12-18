"""
After a backbone model has been trained, we freeze it and use its representations as the
input to a linear classifier. This classifier is trained to predict labels on input data

What needs to happen here is to read in a checkpoint, load up the backbone from this checkpoint, have a standard training loop that runs for n epochs (also read from the command line), have a validation and test set, and save the linear classifier every n epochs

Should use the HDRIFeatureClassificationDataset for the training, validation, and test sets (we should use a 85% train, 5% validation, 10% test split)

Should have linear warmup for the first so many epochs (again from command line), then decay

"""

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import argparse
import os
from src.models.classifiers.linear_classifier import LinearClassifier
from src.datasets.hdri_feature_classification_dataset import HDRIFeatureClassificationDataset
from src.models.backbones.res_net_wrapper import ResNetWrapper
from src.utils.standard import set_seed, get_combined_scheduler, save_checkpoint

def load_backbone_checkpoint(filename, backbone, device='cuda'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
    else:
        raise FileNotFoundError(f"No backbone checkpoint found at '{filename}'")

def load_classifier_checkpoint(filename, classifier, device='cuda'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    else:
        raise FileNotFoundError(f"No classifier checkpoint found at '{filename}'")

def train_classifier(
    backbone,
    classifier,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device='cuda',
    checkpoint_dir=None
):
    backbone.eval()  # Set backbone to eval mode
    classifier.to(device)
    
    for epoch in range(num_epochs):
        # TRAIN
        classifier.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                representations = backbone(images).squeeze()
            
            optimizer.zero_grad()
            outputs = classifier(representations)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss/(batch_idx+1):.4f}'})
        scheduler.step()

        # VALIDATE
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                representations = backbone(images).squeeze()
                outputs = classifier(representations)
                if outputs.shape != labels.shape:
                    outputs = outputs.view(labels.shape)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}')
        
        # SAVE CHECKPOINT
        if checkpoint_dir:
            save_checkpoint({ 
                'epoch': epoch + 1,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            checkpoint_dir,
            epoch + 1,
            loss=val_loss,
            prefix='classifier'
            ) 

def evaluate_model(backbone, classifier, test_loader, criterion, device='cuda'):
    backbone.eval()
    classifier.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            representations = backbone(images).squeeze()
            outputs = classifier(representations)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy

def get_data_loaders(dataset, batch_size, num_workers):
    train_size = int(0.85 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train or Evaluate Linear Classifier on Frozen Backbone')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train', help='Mode: train or evaluate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num_warmup_steps', type=int, default=20, help='Number of warmup steps for scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save classifier checkpoints')
    parser.add_argument('--backbone_checkpoint', type=str, required=True, help='Path to the backbone checkpoint')
    parser.add_argument('--classifier_checkpoint', type=str, help='Path to the classifier checkpoint (required for evaluation)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training/evaluation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoaders')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoaders')
    parser.add_argument('--rendered_image_folder', type=str, required=True, help='Path to rendered image folder')
    parser.add_argument('--hdri_parent_folder', type=str, required=True, help='Path to HDRI parent folder')
    parser.add_argument('--scene_name', type=str, required=True, help='Scene name for the dataset')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Initialize Backbone
    backbone = ResNetWrapper('resnet18')
    load_backbone_checkpoint(args.backbone_checkpoint, backbone, device=device)
    backbone.to(device)
    backbone.eval()
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Prepare Dataset and DataLoaders
    dataset = HDRIFeatureClassificationDataset(
        rendered_image_folder=args.rendered_image_folder,
        hdri_parent_folder=args.hdri_parent_folder,
        scene_name=args.scene_name,
    )
    train_loader, val_loader, test_loader = get_data_loaders(dataset, args.batch_size, args.num_workers)

    if args.mode == 'train':
        # Initialize Classifier
        classifier = LinearClassifier(input_dim=512, num_classes=10)
        classifier.to(device)

        # Define Loss, Optimizer, Scheduler
        criterion = BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
        scheduler = get_combined_scheduler(optimizer, args.num_warmup_steps, args.num_epochs)
        
        # Train the Classifier
        train_classifier(
            backbone,
            classifier,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            args.num_epochs,
            device=device,
            checkpoint_dir=args.checkpoint_dir
        )
    
    elif args.mode == 'evaluate':
        if not args.classifier_checkpoint:
            parser.error("--classifier_checkpoint is required for evaluation mode.")
        
        # Initialize Classifier
        classifier = LinearClassifier(input_dim=512, num_classes=10)
        load_classifier_checkpoint(args.classifier_checkpoint, classifier, device=device)
        classifier.to(device)
        classifier.eval()
        
        # Define Loss
        criterion = BCEWithLogitsLoss()
        
        # Evaluate the Model
        print("Evaluating on test data...")
        evaluate_model(backbone, classifier, test_loader, criterion, device=device)

if __name__ == '__main__':
    main()