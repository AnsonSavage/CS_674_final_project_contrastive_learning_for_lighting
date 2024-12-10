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


def load_backbone_checkpoint(filename, backbone, device=None):
    if os.path.isfile(filename):
        # if the device is CPU, we need to map the storage location to CPU
        map_location = 'cpu' if device == 'cpu' else None
        checkpoint = torch.load(filename, map_location=map_location)
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
    else:
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")


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
        classifier.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad(): # For good measure, but I don't think this is necessary
                representations = backbone(images).squeeze()
            
            optimizer.zero_grad()
            outputs = classifier(representations)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss/(batch_idx+1):.4f}'})
        scheduler.step()


        # Validation loop
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): # The entire validation loop should be done without gradients
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                representations = backbone(images).squeeze()
                outputs = classifier(representations)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f} - Val Accuracy: {accuracy:.2f}%')
        if checkpoint_dir:
            save_checkpoint({  # Updated to use save_checkpoint
                'epoch': epoch + 1,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_dir, epoch + 1, prefix='classifier')  # Added prefix


if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train Linear Classifier on Frozen Backbone')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num_warmup_steps', type=int, default=20, help='Number of warmup steps for scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save classifier checkpoints')
    parser.add_argument('--backbone_checkpoint', type=str, required=True, help='Path to the backbone checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    args = parser.parse_args()
    backbone = ResNetWrapper('resnet18')
    load_backbone_checkpoint(args.backbone_checkpoint, backbone, device=args.device)
    backbone.to(torch.device(args.device))
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    classifier = LinearClassifier(
        input_dim=512,
        num_classes=10
    )

    dataset = HDRIFeatureClassificationDataset(
        rendered_image_folder='/home/ansonsav/cs_674/CS_674_final_project_contrastive_learning_for_lighting/training_data/test_1',
        hdri_parent_folder='/home/ansonsav/cs_674/CS_674_final_project_contrastive_learning_for_lighting/hdris',
        scene_name='lone-monk_cycles_and_exposure-node_demo',
    )
    train_size = int(0.85 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    criterion = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = get_combined_scheduler(optimizer, args.num_warmup_steps, args.num_epochs)
    train_classifier(
        backbone,
        classifier,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        args.num_epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )

