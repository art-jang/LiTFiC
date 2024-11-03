import torch
import torch.nn as nn

class LipEncoder(nn.Module):
    def __init__(self,
                 lip_feat_dim, 
                 out_dim,):
        super(LipEncoder, self).__init__()
       
        self.encoder = nn.Linear(lip_feat_dim, out_dim)

    def forward(self, x, mask=None):
        x = self.encoder(x)
        return x, mask