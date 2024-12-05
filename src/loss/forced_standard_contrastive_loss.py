"""
Essentially the same as the standard contrastive loss where each row_1 != row_2 are forced negative paris and row_1 == row_2 are forced positive pairs.
"""
import torch
# Define a loss function that inherits from torch.nn.Module

class ForcedStandardContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(ForcedStandardContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute the NT-Xent loss between two batches of embeddings.

        Args:
            z_i: Tensor of shape (batch_size, embedding_dim)
            z_j: Tensor of shape (batch_size, embedding_dim)

        Returns:
            loss: Scalar tensor containing the loss
        """
        batch_size = z_i.size(0)

        # Step 1: Normalize the embeddings
        z_i_normalized = torch.nn.functional.normalize(z_i, dim=1)
        z_j_normalized = torch.nn.functional.normalize(z_j, dim=1)

        # Step 2: Concatenate the normalized embeddings
        representations = torch.cat([z_i_normalized, z_j_normalized], dim=0)  # (2*batch_size, embedding_dim)

        # Step 3: Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)  # (2*batch_size, 2*batch_size)

        # Step 4: Create masks to identify positive and negative pairs
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(z_i.device)
        masks = torch.eye(batch_size * 2, dtype=torch.bool).to(z_i.device)

        # Positive mask: Matches where labels are the same but not the same index (excluding self-comparisons)
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~masks)

        # Negative mask: Matches where labels are different
        negative_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))

        # Step 5: Compute logits
        logits = similarity_matrix / self.temperature

        # For numerical stability, subtract max logits
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Step 6: Compute exponential logits
        exp_logits = torch.exp(logits)

        # Compute log probabilities
        log_prob = logits - torch.log(exp_logits * negative_mask.float()).sum(1, keepdim=True)

        # Only keep positive log probabilities
        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(1) / positive_mask.sum(1)

        # Loss is the negative of mean log probability
        loss = -mean_log_prob_pos.mean()

        return loss