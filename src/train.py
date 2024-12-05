import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import random
import argparse
from models.projection_heads.sim_clr_projection_head import SimCLRProjectionHead
from models.backbones.res_net_wrapper import ResNetWrapper
from datasets.contrastive_hdri_complete_sample_dataset import ContrastiveHDRIDataset

def set_seed(seed):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def train_model(backbone, projection_head, train_loader, criterion, optimizer, scheduler, num_epochs, device='cuda'):
    """
    Standard training loop for PyTorch model
    """
    # Move model to device
    backbone = backbone.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        backbone.train()
        running_loss = 0.0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images_1, images_2) in enumerate(pbar):
            # Move data to device
            images_1 = images_1.to(device)
            images_2 = images_2.to(device)

            complete_batch = torch.cat((images_1, images_2), dim=0) # Currently, the two contrastive losses get concatenated together so that they only have to pass through the network once
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            representations = backbone(complete_batch) # h_i
            representations = representations.squeeze() # Remove extra dimensions
            projections = projection_head(representations) # z_i

            assert projections.shape[0] % 2 == 0, "Projections must be split evenly"
            projections_1, projections_2 = torch.split(projections, projections.shape/2, dim=0)

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
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}')
    
if __name__ == '__main__':
    set_seed(42)
    
    # NOTE: After going through the ResNetWrapper, both input images of size (3, 224, 224) and (3, 256, 256) come out to be (512, 1, 1) in size. I'm not sure if it resizes it internally, but it looks like keeping it at our default size of (3, 256, 256) is fine.
    
    # Example usage with CosineAnnealingLR:
    backbone = ResNetWrapper('resnet18')
    projection_head = SimCLRProjectionHead()
    dataset = ContrastiveHDRIDataset(
        image_folder='/home/ansonsav/cs_674/CS_674_final_project_contrastive_learning_for_lighting/training_data/test_1',
        scene_name='lone-monk_cycles_and_exposure-node_demo',
        )
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4
    )

    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    # num_epochs = 10

    # trained_model = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)

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