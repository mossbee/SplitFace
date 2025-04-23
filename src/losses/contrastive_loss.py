from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Normalize the feature vectors
        features = nn.functional.normalize(features, dim=1)

        # Compute the similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create labels for positive and negative pairs
        labels = labels.unsqueeze(1) == labels.unsqueeze(0)
        labels = labels.float()

        # Compute the contrastive loss
        positive_pairs = similarity_matrix * labels
        negative_pairs = similarity_matrix * (1 - labels)

        # Calculate the loss
        loss = -torch.log(positive_pairs.sum(dim=1) / (positive_pairs.sum(dim=1) + negative_pairs.sum(dim=1) + 1e-8))

        return loss.mean()