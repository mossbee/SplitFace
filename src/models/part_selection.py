from torch import nn
import torch

class PartSelectionModule(nn.Module):
    def __init__(self, num_patches, num_selected_patches):
        super(PartSelectionModule, self).__init__()
        self.num_patches = num_patches
        self.num_selected_patches = num_selected_patches
        self.attention_weights = nn.Linear(num_patches, num_patches)

    def forward(self, x):
        # x is expected to be of shape (batch_size, num_patches, embed_dim)
        attention_scores = self.attention_weights(x)  # Shape: (batch_size, num_patches, num_patches)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Shape: (batch_size, num_patches, num_patches)

        # Select the top K patches based on attention weights
        top_k_indices = torch.topk(attention_weights, self.num_selected_patches, dim=-1).indices  # Shape: (batch_size, num_patches, num_selected_patches)

        selected_patches = []
        for i in range(top_k_indices.size(0)):
            selected_patches.append(x[i, top_k_indices[i]])  # Select patches for each sample in the batch

        return torch.stack(selected_patches)  # Shape: (batch_size, num_selected_patches, embed_dim)