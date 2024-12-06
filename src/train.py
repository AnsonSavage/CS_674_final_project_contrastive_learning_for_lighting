import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import random
import argparse
from models.projection_heads.sim_clr_projection_head import SimCLRProjectionHead
from models.backbones.res_net_wrapper import ResNetWrapper
from datasets.contrastive_hdri_complete_sample_dataset import ContrastiveHDRIDataset
from loss.forced_standard_contrastive_loss import ForcedStandardContrastiveLoss

def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_combined_scheduler(
    optimizer: torch.optim.Optimizer, 
    num_warmup_steps: int, 
    num_epochs: int
) -> torch.optim.lr_scheduler.SequentialLR:
    """
    Create a combined learning rate scheduler with a warm-up phase followed by cosine annealing.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warm-up phase.
        num_epochs (int): The total number of training epochs.

    Returns:
        torch.optim.lr_scheduler.SequentialLR: The combined scheduler.
    """
    def get_scheduler_with_warmup(
        optimizer: torch.optim.Optimizer, 
        num_warmup_steps: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Create a LambdaLR scheduler for the warm-up phase.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
            num_warmup_steps (int): The number of steps to linearly increase the learning rate.

        Returns:
            torch.optim.lr_scheduler.LambdaLR: The warm-up scheduler.
        """
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            raise ValueError(f"Invalid step: {current_step} for warmup scheduler")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    warmup_scheduler = get_scheduler_with_warmup(optimizer, num_warmup_steps)
    assert num_epochs > num_warmup_steps, "Number of epochs must be greater than the number of warm-up steps"
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - num_warmup_steps, eta_min=0)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])

def train_model(
    backbone: torch.nn.Module,
    projection_head: torch.nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int,
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Train the model using the provided components.

    This function handles the training loop, including forward and backward passes, loss computation,
    optimizer steps, and scheduler updates.

    Args:
        backbone (torch.nn.Module): The backbone neural network model.
        projection_head (torch.nn.Module): The projection head network.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion: Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train the model.
        device (str, optional): Device to run the training on. Defaults to 'cuda'.

    Returns:
        torch.nn.Module: The trained backbone model.
    """
    # Move model to device
    backbone = backbone.to(device)
    projection_head = projection_head.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
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
    
if __name__ == '__main__':
    set_seed(42)
    
    # NOTE: After going through the ResNetWrapper, both input images of size (3, 224, 224) and (3, 256, 256) come out to be (512, 1, 1) in size. I'm not sure if it resizes it internally, but it looks like keeping it at our default size of (3, 256, 256) is fine.
    
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
        lr=0.001
    )
    
    num_epochs = 1000
    num_warmup_steps = 2

    scheduler = get_combined_scheduler(optimizer, num_warmup_steps, num_epochs)

    trained_model = train_model(backbone, projection_head, train_loader, criterion, optimizer, scheduler, num_epochs, device='cuda')

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