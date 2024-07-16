import re
import torch
import torch.nn as nn

class MMProjector(nn.Module):
    def __init__(self,
                 projector_type,
                 mm_hidden_size,
                 hidden_size,
                 precision,
                 **kwargs):
        super(MMProjector, self).__init__()
        self.projector_type = projector_type

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(mm_hidden_size, hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(hidden_size, hidden_size))
            self.projector = nn.Sequential(*modules)
        else:
            if projector_type == 'linear':
                self.projector = nn.Linear(mm_hidden_size, hidden_size)
            else:
                raise ValueError(f'Unknown projector type: {projector_type}')
        
    def forward(self, x):
        x = self.projector(x)
        return x
