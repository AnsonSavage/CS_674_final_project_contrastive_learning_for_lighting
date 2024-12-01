import torch
from tqdm import tqdm
import torch.nn as nn

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device='cuda'):
    """
    Standard training loop for PyTorch model
    """
    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{running_loss/(batch_idx+1):.4f}'})
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader)
        
        # Step scheduler with epoch loss if it's ReduceLROnPlateau, otherwise step normally
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}')
    
    return model

# Example usage:
"""
model = YourModel()
train_loader = DataLoader(...)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
num_epochs = 10

trained_model = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)
"""

# Example usage with CosineAnnealingLR:
model = YourModel()
train_loader = DataLoader(...)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

num_epochs = 10

trained_model = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)
