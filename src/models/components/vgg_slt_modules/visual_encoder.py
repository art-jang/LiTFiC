import torch
import torch.nn as nn

class VisualEncoder(nn.Module):
    def __init__(self,
                 encoder_type,
                 out_dim,
                 load_features=False,
                 precision="float32",
                 **kwargs):
        super(VisualEncoder, self).__init__()
        self.encoder_type = encoder_type

        if load_features:
            self.visual_encoder = nn.Identity()

    def forward(self, x, mask=None):
        x = self.visual_encoder(x)
        return x
