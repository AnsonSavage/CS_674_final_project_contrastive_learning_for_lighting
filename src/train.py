import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from models.projection_heads.sim_clr_projection_head import SimCLRProjectionHead
from models.backbones.res_net_wrapper import ResNetWrapper
from datasets.contrastive_hdri_complete_sample_dataset import ContrastiveHDRIDataset
from loss.forced_standard_contrastive_loss import ForcedStandardContrastiveLoss
from utils.standard import set_seed, get_combined_scheduler, save_checkpoint


def load_checkpoint(filename: str, backbone: torch.nn.Module, projection_head: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler) -> int:
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    return 0

def train_model(
    backbone: torch.nn.Module,
    projection_head: torch.nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int,
    device: str = 'cuda',
    checkpoint_dir: str = None,
    starting_checkpoint: str = None
) -> torch.nn.Module:
    """
    Train the model using the provided components.

    This function handles the training loop, including forward and backward passes, loss computation,
    optimizer steps, scheduler updates, and periodic callbacks.

    Args:
        backbone (torch.nn.Module): The backbone neural network model.
        projection_head (torch.nn.Module): The projection head network.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion: Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train the model.
        device (str, optional): Device to run the training on. Defaults to 'cuda'.
        checkpoint_dir (str, optional): Directory to save checkpoints. If None, checkpointing is disabled.
        starting_checkpoint (str, optional): Path to a starting checkpoint. Defaults to None.

    Returns:
        torch.nn.Module: The trained backbone model.
    """
    # Move model to device
    backbone = backbone.to(device)
    projection_head = projection_head.to(device)
    
    start_epoch = load_checkpoint(starting_checkpoint, backbone, projection_head, optimizer, scheduler) if checkpoint_dir is None else 0
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        backbone.train()
        projection_head.train()
        running_loss = 0.0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images_1, images_2) in enumerate(pbar):
            # Move data to device
            images_1 = images_1.to(device).squeeze(0) # NOTE: In current implementation, the dataloader automatically loads a batch of 12, and so we are squeezing the batch dimension
            images_2 = images_2.to(device).squeeze(0)

            complete_batch = torch.cat((images_1, images_2), dim=0) # Currently, the two contrastive losses get concatenated together so that they only have to pass through the network once
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            representations = backbone(complete_batch) # h_i, (2*batch_size, num_channels, 1, 1)
            representations = representations.squeeze() # After convolutional layers, the output is (2*batch_size, num_channels, 1, 1), so we squeeze the last two dimensions, (2*batch_size, num_channels)
            projections = projection_head(representations) # z_i, (2*batch_size, projection_dim)

            assert projections.shape[0] % 2 == 0, "Projections must be split evenly"
            projections_1, projections_2 = torch.chunk(projections, 2, dim=0)

            loss = criterion(projections_1, projections_2) # Self supervised loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{running_loss/(batch_idx+1):.4f}'})

        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader)
        
        scheduler.step()

        # I'm not sure if these should return the same learning rate or not, this is an experiment
        print("Learning rate from optimizer: ", optimizer.param_groups[0]['lr'])
        print("Learning rate from scheduler: ", scheduler.get_last_lr())
        # assert scheduler.get_last_lr() == optimizer.param_groups[0]['lr'], "Learning rate mismatch"
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Learning Rate: {optimizer.param_groups[0]["lr"]}')
        
        # Save checkpoint at the end of each epoch if checkpoint_dir is provided
        if checkpoint_dir:
            save_checkpoint({
                'epoch': epoch + 1,
                'backbone_state_dict': backbone.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_dir, epoch + 1, loss=epoch_loss)
    
if __name__ == '__main__':
    set_seed(42)
    
    # NOTE: After going through the ResNetWrapper, both input images of size (3, 224, 224) and (3, 256, 256) come out to be (512, 1, 1) in size. I'm not sure if it resizes it internally, but it looks like keeping it at our default size of (3, 256, 256) is fine.
    
    import argparse

    parser = argparse.ArgumentParser(description='Train Contrastive Learning Model')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--num_warmpup_steps', type=int, default=10, help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    args = parser.parse_args()

    backbone = ResNetWrapper('resnet18')
    projection_head = SimCLRProjectionHead()
    dataset = ContrastiveHDRIDataset(
        image_folder='/home/ansonsav/cs_674/CS_674_final_project_contrastive_learning_for_lighting/training_data/test_1',
        scene_name='lone-monk_cycles_and_exposure-node_demo',
        total_batches=10,
        )
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4
    )

    criterion = ForcedStandardContrastiveLoss()
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(projection_head.parameters()), # We want to optimize both the backbone and the projection head. len(list(backbone.parameters())) is 62 for resnet18
        lr=args.learning_rate
    )
    
    num_epochs = args.num_epochs
    num_warmup_steps = args.num_warmpup_steps

    scheduler = get_combined_scheduler(optimizer, num_warmup_steps, num_epochs)

    trained_model = train_model(
        backbone, 
        projection_head, 
        train_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs, 
        device='cuda',
        checkpoint_dir=args.checkpoint_dir,
    )

"""
When running this code:

for x, y in train_loader:
    print('before')
    x, y = x.squeeze(0), y.squeeze(0)
    print(x.shape, y.shape)

    print('after backbone')
    backbone_output = backbone(x)
    print(backbone_output.shape)
    backbone_output = backbone_output.squeeze()
    print('after squeeze')
    print(backbone_output.shape)
    print('after projection head')
    print(projection_head(backbone_output).shape)
    break

I get the following output:

before
torch.Size([12, 3, 256, 256]) torch.Size([12, 3, 256, 256])
after backbone
torch.Size([12, 512, 1, 1])
after squeeze
torch.Size([12, 512])
after projection head
torch.Size([12, 128])

"""