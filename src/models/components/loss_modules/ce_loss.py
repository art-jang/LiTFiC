import torch
import torch.nn as nn

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        logits = preds["logits"]
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        return loss
