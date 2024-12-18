import re
import torch
import torch.nn as nn


class MMProjector(nn.Module):
    def __init__(self,
                 projector_type,
                 mm_hidden_size,
                 hidden_size,
                 dropout=0.0,
                 **kwargs):
        super(MMProjector, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.projector_type = projector_type

        self.mapping = None
        
        self.projector_type = projector_type
        # mapping network
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
            elif 'conv' in projector_type:
                conv_match = re.match(r'^conv_K(\d+)_P(\d+)_KP(\d+)_L(\d+)$', projector_type)
                assert conv_match, f'Invalid projector type: {projector_type}'
                
                self.kernel_size = int(conv_match.group(1))
                self.stride = int(conv_match.group(2))
                self.kernel_size_pool = int(conv_match.group(3))
                self.num_layers = int(conv_match.group(4))
                
                projector = []
                for _ in range(self.num_layers):
                    projector.append(nn.Conv1d(mm_hidden_size, mm_hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size//2))
                    projector.append(nn.ReLU())
                    projector.append(nn.MaxPool1d(kernel_size=self.kernel_size_pool, stride=self.stride))
                self.projector = nn.Sequential(*projector)
                self.projector2 = nn.Linear(mm_hidden_size, hidden_size)
            else:
                raise ValueError(f'Unknown projector type: {projector_type}')


    def forward(self, x, masks=None):
        if 'conv' in self.projector_type:
            x = x.permute(0, 2, 1)
        x = self.projector(x)
        if 'conv' in self.projector_type:
            x = x.permute(0, 2, 1)
            for i in range(self.num_layers):
                # if i != self.num_layers -1:
                masks = torch.nn.functional.max_pool1d(masks, kernel_size=self.kernel_size_pool, stride=self.stride)
            x = x * masks.unsqueeze(-1)
            x = self.projector2(x)
            
        x = self.dropout(x)
        
        return x, masks