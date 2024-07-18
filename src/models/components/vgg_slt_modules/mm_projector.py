import re
import torch
import torch.nn as nn

class MMProjector(nn.Module):
    def __init__(self,
                 projector_type,
                 mm_hidden_size,
                 hidden_size,
                 pooling_config,
                 **kwargs):
        super(MMProjector, self).__init__()
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

    def forward(self, x, masks=None):
        if self.early_fusion:
            x, masks = self._pool(x, masks)
            x = self.projector(x)
        else:
            x = self.projector(x)
            x, masks = self._pool(x, masks)
        return x, masks
