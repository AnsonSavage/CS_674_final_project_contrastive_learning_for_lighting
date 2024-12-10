import random
import torch
import os
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
        num_warmup_steps: int: The number of steps for the warm-up phase.
        num_epochs: int: The total number of training epochs.

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
            optimizer: torch.optim.Optimizer: The optimizer for which to schedule the learning rate.
            num_warmup_steps: int: The number of steps to linearly increase the learning rate.

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

def save_checkpoint(state: dict, checkpoint_dir: str, epoch: int, loss: float = None, prefix: str = None) -> None:
    """
    Save the training checkpoint.

    Args:
        state (dict): State dictionary containing model and optimizer states.
        checkpoint_dir (str): Directory to save the checkpoint.
        epoch (int): Current epoch number.
        loss (float, optional): Epoch loss to include in the filename. Defaults to None.
        prefix (str, optional): Prefix to prepend to the checkpoint filename. Defaults to None.
    """
    if prefix:
        prefix = f'{prefix}_'
    else:
        prefix = ''
    if loss is not None:
        filename = os.path.join(checkpoint_dir, f'{prefix}checkpoint_epoch_{epoch}_loss_{loss:.4f}.pth')
    else:
        filename = os.path.join(checkpoint_dir, f'{prefix}checkpoint_epoch_{epoch}.pth')
    torch.save(state, filename)