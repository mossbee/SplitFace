import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.contrastive_loss import ContrastiveLoss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.contrastive_loss = ContrastiveLoss()

    def forward(self, outputs, targets, part_tokens):
        classification_loss = F.cross_entropy(outputs, targets)
        contrastive_loss = self.contrastive_loss(part_tokens, targets)
        return self.alpha * classification_loss + (1 - self.alpha) * contrastive_loss