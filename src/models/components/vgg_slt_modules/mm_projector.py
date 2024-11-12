import re
import torch
import torch.nn as nn
import hydra

import ipdb

from src.utils.cslr_tools import load_checkpoint_model
from omegaconf import OmegaConf
from src.models.components.vgg_slt_modules.qformer import QFormer

class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)
    
class MMProjector(nn.Module):
    def __init__(self,
                 cslr2_config,
                 use_qformer,
                 qformer_config,
                 projector_type,
                 mm_hidden_size,
                 hidden_size,
                 pooling_config,
                 cslr2_options,
                 dropout=0.0,
                 **kwargs):
        super(MMProjector, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.projector_type = projector_type
        self.use_cslr2 = cslr2_options.use

        self.mapping = None
        if self.use_cslr2:
            if mm_hidden_size != 768:
                self.mapping = nn.Linear(mm_hidden_size, 768)
                mm_hidden_size = 768
            self.cslr2 = cslr2_config

            if cslr2_options.ckpt_path is not None:
                self.cslr2 = load_checkpoint_model(cslr2_options.ckpt_path, self.cslr2, 'cpu')
                
            if cslr2_options.freeze:
                for name, param in self.cslr2.named_parameters():
                    param.requires_grad = False
        
        self.use_qformer = use_qformer
        if use_qformer:
            self.qformer = QFormer(**qformer_config)
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
                for i in range(self.num_layers):
                    projector.append(nn.Conv1d(mm_hidden_size, mm_hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size//2))
                    if False:
                        if True:
                            projector.append(nn.BatchNorm1d(mm_hidden_size))
                        else:
                            projector.append(Permute(0, 2, 1))
                            projector.append(nn.LayerNorm(mm_hidden_size))
                            projector.append(Permute(0, 2, 1))
                    projector.append(nn.ReLU())
                    # if i != self.num_layers - 1:
                    projector.append(nn.MaxPool1d(kernel_size=self.kernel_size_pool, stride=self.stride))
                self.projector = nn.Sequential(*projector)
                self.projector2 = nn.Linear(mm_hidden_size, hidden_size)
            else:
                raise ValueError(f'Unknown projector type: {projector_type}')

        # adaptive pooling
        self.early_fusion = pooling_config.early_fusion
        self.adaptive_pool = pooling_config.adaptive_pool
        if self.adaptive_pool:
            self.depth = pooling_config.depth
            aggregation_dim = mm_hidden_size if self.early_fusion else hidden_size
            modules = [torch.nn.TransformerEncoderLayer(
                                    d_model=aggregation_dim,
                                    nhead=pooling_config.nhead,
                                    dim_feedforward=aggregation_dim*2,
                                    activation=pooling_config.activation,
                                    batch_first=pooling_config.batch_first,
                                ) for _ in range(pooling_config.depth)]
            modules.append(nn.Linear(aggregation_dim, 1))
            modules.append(nn.Sigmoid())
            self.aggregator = nn.Sequential(*modules)
        else:
            self.aggregator = None

    def _pool(self, x, masks):
        # x: [B, T, C]
        # masks: [B, T]
        if self.adaptive_pool:
            pooled_x = []
            new_masks = []
            
            for i in range(self.depth):
                feat_pool = self.aggregator[i](x, src_key_padding_mask=masks)

            max_pooled_len = 0
            for b in range(x.size(0)):
                valid_len = int(masks[b].sum())
                valid_feat = feat_pool[b:b+1, :valid_len]
                valid_feat = self.aggregator[i:](valid_feat)

                # Apply sigmoid to the logits
                threshold = 1.0
                cumsum = torch.cumsum(valid_feat, dim=1) // threshold
                
                pooled_len = int(cumsum.max().item()) + 1
                scatter_mat = torch.zeros((pooled_len, len(cumsum[0]))).to(x.device).scatter_(0, cumsum.squeeze(-1).long(), 1)
                scatter_mat = scatter_mat.to(x.dtype)
                weight_mat = scatter_mat * valid_feat[0].T
                final_feat = weight_mat @ x[b, :valid_len]
                pooled_x.append(final_feat)
                if final_feat.size(0) > max_pooled_len:
                    max_pooled_len = final_feat.size(0)
            
            # Padding
            for i in range(len(pooled_x)):
                padded_len = max_pooled_len - pooled_x[i].size(0)
                mask = torch.cat([torch.ones(pooled_x[i].size(0)), torch.zeros(padded_len)]).to(x.device)
                new_masks.append(mask)
                if padded_len > 0:
                    padded = torch.zeros((padded_len, pooled_x[i].size(1))).to(x.device)
                    pooled_x[i] = torch.cat([pooled_x[i], padded], dim=0)
            pooled_x = torch.stack(pooled_x, dim=0) # [B, T', C]
            new_masks = torch.stack(new_masks, dim=0) # [B, T']
        else:
            pooled_x = x
            new_masks = masks
            
        return pooled_x, new_masks

    def forward(self, x, masks=None, target_indices = None, target_labels = None):
        if self.use_cslr2:
            if self.mapping is not None:
                x = self.mapping(x)
            x, masks = self.cslr2(x, masks, target_indices, target_labels)
        
        if self.use_qformer:
            x = self.qformer(x, masks)
            masks = None

            return x, masks
            
        # if self.early_fusion:
        #     x, masks = self._pool(x, masks)
        #     x = self.projector(x)
        # else:
        #     x = self.projector(x)
        #     x, masks = self._pool(x, masks)
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