"""
Essentially the same as the standard contrastive loss where each row_1 != row_2 are forced negative paris and row_1 == row_2 are forced positive pairs.
"""
import torch
# TODO: Step through all of the math and convince yourself that it is correct

class ForcedStandardContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature = 0.1):
        # In the paper, a temperature of 0.1 yielded the highest performance
        super(ForcedStandardContrastiveLoss, self).__init__()
        self.temperature = temperature # Temperature values greater than 1 will smooth the distribution and values less than 1 will sharpen the distribution

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
        # This is done so that when they are dot-producted using torch.matmul, the cosine similarity is computed (because they will be of unit length)
        z_i_normalized = torch.nn.functional.normalize(z_i, dim=1)
        z_j_normalized = torch.nn.functional.normalize(z_j, dim=1)

        # Step 2: Concatenate the normalized embeddings
        # So, now we have a big stack of the embeddings where positive pairs correspond via (index + batch_size) % (2*batch_size)
        representations = torch.cat([z_i_normalized, z_j_normalized], dim=0)  # (2*batch_size, embedding_dim)

        # Step 3: Compute similarity matrix
        # This is now a symmetric matrix where index (i, j) corresponds to the cosine similarity between the ith and jth embeddings
        # This means that the diagonal will all be 1s b/c i=j
        similarity_matrix = torch.matmul(representations, representations.T)  # (2*batch_size, 2*batch_size)

        # Step 4: Create masks to identify positive and negative pairs
        # labels is a 1D tensor where the first batch_size elements are 0, 1, 2, ..., batch_size-1 and the second batch_size elements are 0, 1, 2, ..., batch_size-1
        # masks is a 2D tensor where the diagonal is all False and the rest is True
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(z_i.device)  # (2*batch_size,)
        masks = ~torch.eye(batch_size * 2, dtype=torch.bool).to(z_i.device)  # (2*batch_size, 2*batch_size)

        # Positive mask: Matches where labels are the same but not the same index (excluding self-comparisons)
        # See the example below for a better understanding
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (masks)

        # Negative mask: Matches where labels are different
        # Note that not only is the diagonal excluded, but also the positive pairs (unlike in the original SimCLR loss). In my mind, it makes sense to not include the positive pairs in the denominator because they would only make it bigger
        # If we wanted it to be the same, we would simply create a mask where the diagonal is False and the rest is True
        negative_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))

        # Step 5: Compute logits
        # Scale the cosine similarity by the temperature (tau)
        # This will either sharpen or smooth the distribution (values greater than 1 will make the distribution smoother, meaning smoother gradients and that negative pairs will not be forced apart as strongly)
        logits = similarity_matrix / self.temperature

        # For numerical stability, subtract max logits
        # Shifting the value of the logits changes absolutely nothing because the principle of softmax has to do with the relative values of the logits (see https://www.desmos.com/calculator/awl0wrvzbm, for example)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Step 6: Compute exponential logits
        # Exponentiate the logits, giving us the main values that will be used in both the numerator and denominator
        exp_logits = torch.exp(logits)

        
        # The rows represent i and the columns represent k. So, each row corresponds to the i part of the (i, j) loss, but we still loop over all k and sum the negative pair cosine similarities
        # By applying the negative_mask, we have excluded all i == k pairs and all i == j pairs
        sum_exp_negatives = (exp_logits * negative_mask.float()).sum(1, keepdim=True)

        # Compute log probabilities
        # Perform subtraction in the log space (the logits remain in their original state because log(exp(x)) = x)
        # This is equivalent to performing a log of exp_logits/sum_exp_negatives
        # TODO: after torch.log(), any entries that are 0 become -inf. If you subtract -inf from any real number, you get inf. How is this handled?
        # Oh, but maybe we won't have any zero values because we did a sum along the rows
        log_prob = logits - torch.log(sum_exp_negatives)

        # Only keep positive log probabilities
        # Because we are only defining the loss for positive pairs (i, j), we mask out all the other pairs that aren't positive. Thus, all the non-zero entries after multiplication with the positive_mask represent this loss for those positive pairs
        # By summing along dimension 1, we reduce this (there is only one non-zero entry per row)
        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(1) # / positive_mask.sum(1) # NOTE: if your code changes such that there maybe is more than one positive pair, you should divide by the number of positive pairs and uncomment this part

        # Loss is the negative of mean log probability
        loss = -mean_log_prob_pos.mean()

        return loss


"""
A couple of examples:
>>> batch_size = 3
>>> labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
>>> labels
tensor([0, 1, 2, 0, 1, 2])
>>> masks = ~torch.eye(batch_size * 2, dtype=torch.bool)
>>> masks
tensor([[False,  True,  True,  True,  True,  True],
        [ True, False,  True,  True,  True,  True],
        [ True,  True, False,  True,  True,  True],
        [ True,  True,  True, False,  True,  True],
        [ True,  True,  True,  True, False,  True],
        [ True,  True,  True,  True,  True, False]])

>>> labels.unsqueeze(0)
tensor([[0, 1, 2, 0, 1, 2]])
>>> labels.unsqueeze(1)
tensor([[0],
        [1],
        [2],
        [0],
        [1],
        [2]])

# Basically, the dimensions are broadcasted so that it'd be like comparing versions of the above in full matrix form

>>> labels.unsqueeze(0) == labels.unsqueeze(1) 
tensor([[ True, False, False,  True, False, False],
        [False,  True, False, False,  True, False],
        [False, False,  True, False, False,  True],
        [ True, False, False,  True, False, False],
        [False,  True, False, False,  True, False],
        [False, False,  True, False, False,  True]])

# Here, we simply exclude the diagonal because we don't want to compare the same embeddings
>>> positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (masks)
>>> positive_mask
tensor([[False, False, False,  True, False, False],
        [False, False, False, False,  True, False],
        [False, False, False, False, False,  True],
        [ True, False, False, False, False, False],
        [False,  True, False, False, False, False],
        [False, False,  True, False, False, False]])


# The negative mask is similar to the positive mask, but it's true wherever the labels are different
>>> negative_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))
>>> negative_mask
tensor([[False,  True,  True, False,  True,  True],
        [ True, False,  True,  True, False,  True],
        [ True,  True, False,  True,  True, False],
        [False,  True,  True, False,  True,  True],
        [ True, False,  True,  True, False,  True],
        [ True,  True, False,  True,  True, False]])
"""