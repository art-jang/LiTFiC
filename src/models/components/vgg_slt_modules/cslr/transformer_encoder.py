"""
Custom Transformer Encoder model.
"""
import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _xavier_uniform(module: nn.Module):
    """
    Xavier uniform initialization for the weights of a module.

    Args:
        module (nn.Module): module to initialize.
    """
    for _, params in module.named_parameters():
        if params.dim() > 1:
            nn.init.xavier_uniform_(params)


def clones(
    module: nn.Module,
    nb_clones: int,
) -> nn.ModuleList:
    """
    Produce nb_clones identical layers.

    Args:
        module (nn.Module): module to clone.
        nb_clones (int): number of clones.

    Returns:
        nn.ModuleList: list of cloned modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(nb_clones)])


class Encoder(nn.Module):
    """Encoder defined as a stack of N layers."""
    def __init__(
        self,
        layer: nn.Module,
        N: int,
        final_norm: bool = True,
    ) -> None:
        """
        Args:
            layer (nn.Module): layer to use.
            N (int): number of layers.
            final_norm (bool): whether to apply layer normalization at the end of the encoder.
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        if final_norm:
            self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        """
        Forward function of the encoder module.
        Pass the input (and mask) through each layer in turn.

        Args:
            x (torch.tensor): input tensor.
            mask (Optional[torch.tensor]): tensor of masks.

        Returns:
            torch.tensor: output tensor.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return (self.norm(x) if hasattr(self, 'norm') else x)


class LayerNorm(nn.Module):
    """LayerNorm module."""
    def __init__(
        self,
        size: List[int],
        eps: float = 1e-6,
    ) -> None:
        """
        Args:
            size (List[int]): size of the layer.
            eps (float): epsilon value for numerical stability.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(
        self,
        x: torch.tensor,
    ) -> torch.tensor:
        """Perform layer normalisation on the input tensor."""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Residual connection followed by a layer norm."""
    def __init__(
        self,
        size: List[int],
        dropout: float,
    ) -> None:
        """
        Args:
            size (List[int]): size of the layer.
            dropout (float): dropout rate.
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.tensor,
        sublayer: nn.Module,
    ):
        """
        Apply residual connection to any sublayer with the same size.

        Args:
            x (torch.tensor): input tensor.
            sublayer (nn.Module): sublayer to apply.

        Returns:
            torch.tensor: output tensor.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder layer module"""
    def __init__(
        self,
        size: List[int],
        self_attention: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ) -> None:
        """
        Args:
            size (List[int]): size of the layer.
            self_attention (nn.Module): self-attention module.
            feed_forward (nn.Module): feed-forward module.
            dropout (float): dropout rate.
        """
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        """Perform forward pass on the input tensor."""
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    """Multi-headed attention module."""
    def __init__(
        self,
        h: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            h (int): number of heads.
            d_model (int): size of the model.
            dropout (float): dropout rate.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # we assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(
        query: torch.tensor,
        key: torch.tensor,
        value: torch.tensor,
        mask: Optional[torch.tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ):
        """
        Compute scaled dot product attention.

        Args:
            query (torch.tensor): query tensor.
            key (torch.tensor): key tensor.
            value (torch.tensor): value tensor.
            mask (Optional[torch.tensor]): tensor of masks.
            dropout (Optional[nn.Dropout]): dropout module.

        Returns:
            output tensor and attention weights.
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.float()
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(
        self,
        query: torch.tensor,
        key: torch.tensor,
        value: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        """
        Forward function of the multi-headed attention module.

        Args:
            query (torch.tensor): query tensor.
            key (torch.tensor): key tensor.
            value (torch.tensor): value tensor.
            mask (Optional[torch.tensor]): tensor of masks.

        Returns:
            torch.tensor: output tensor.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for l, x in zip(self.linears, (query, key, value))]

        # 2) apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionWiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            d_model (int): size of the model.
            d_ff (int): size of the feed-forward layer.
            dropout (float): dropout rate.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Perform forward pass on the input tensor."""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """Positional Encoding module."""
    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_len: int = 5000,
    ) -> None:
        """
        Args:
            d_model (int): size of the model.
            dropout (float): dropout rate.
            max_len (int): maximum length of the sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Perform positional encoding on the input tensor."""
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Embeddings(nn.Module):
    """Embedding module."""
    def __init__(
        self,
        d_model: int,
        vocab: int,
    ) -> None:
        """
        Args:
            d_model (int): size of the model.
            vocab (int): size of the vocabulary.
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Perform forward pass on the input tensor using the embedding layer."""
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    """Generator module (linear projection in vocab space)."""
    def __init__(
        self,
        d_model: int,
        vocab: int,
    ) -> None:
        """
        Args:
            d_model (int): size of the model.
            vocab (int): size of the vocabulary.
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Project the input tensor into the vocabulary space."""
        return self.proj(x)


class EncoderWithLinear(nn.Module):
    """Encoder with linear projection."""
    def __init__(
        self,
        encoder: Encoder,
        generator: Generator,
        src_embed: Embeddings,
        d_model: int = 768,
        contrastive: bool = False,
    ) -> None:
        """
        Args:
            encoder (Encoder): encoder module.
            generator (Generator): generator module.
            src_embed (Embeddings): positional encoding module.
            d_model (int): size of the model.
            contrastive (bool): whether to use contrastive learning.
        """
        super(EncoderWithLinear, self).__init__()
        self.encoder = encoder
        self.generator = generator
        self.src_embed = src_embed
        self.contrastive = contrastive
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) if self.contrastive else None

    def forward(
        self,
        src: Optional[torch.tensor] = None,
        src_mask: Optional[torch.tensor] = None,
    ):
        """
        Forward function of the encoder module.

        Args:
            src (Optional[torch.tensor]): input tensor.
            src_mask (Optional[torch.tensor]): tensor of masks.

        Returns:
            If contrastive, return the encoder output
            and the linear projection of the encoder output.
            Else return the linear projection of the encoder output.
        """
        encoder_out, src_mask = self.encode(src, src_mask)
        if self.contrastive:
            return encoder_out, self.generator(encoder_out[:, 1:, :])
        return self.generator(encoder_out)

    def encode(
        self,
        src: torch.tensor,
        src_mask: torch.tensor,
    ):
        """
        Encode the input tensor.

        Args:
            src (torch.tensor): input tensor.
            src_mask (torch.tensor): tensor of masks.

        Returns:
            encoder output and mask.
        """
        if isinstance(src, tuple):
            src, src_mask = src
        src_embeddings = self.src_embed(src)
        if self.contrastive:
            src_embeddings = torch.cat(
                [
                    self.cls_token.expand(src_embeddings.size(0), -1, -1),
                    src_embeddings
                ],
                dim=1,
            )
            if src_mask is not None:
                src_mask = torch.cat(
                    [
                        torch.ones(src_mask.size(0), 1).to(src_mask.device),
                        src_mask
                    ],
                    dim=1,
                )
        return self.encoder(src_embeddings, src_mask), src_mask


def make_model(
    vocab: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
    contrastive: bool = False,
) -> EncoderWithLinear:
    """
    Function to create an instance of the EncoderWithLinear model.

    Args:
        vocab (int): size of the vocabulary.
        N (int): number of layers.
        d_model (int): size of the model.
        d_ff (int): size of the feed-forward layer.
        h (int): number of heads.
        dropout (float): dropout rate.
        contrastive (bool): whether to use contrastive learning.

    Returns:
        EncoderWithLinear: instance of the EncoderWithLinear model.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderWithLinear(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        generator=Generator(d_model, vocab),
        src_embed=position,
        d_model=d_model,
        contrastive=contrastive,
    )
    # initialise parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
