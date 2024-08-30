import torch
import torch.nn as nn
import math
import ipdb

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_length, embedding_dim):
        super(LearnablePositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_length, embedding_dim)

    def forward(self, positions):
        return self.embedding(positions)

class FourierPositionalEmbedding(nn.Module):
    def __init__(self, max_length, embedding_dim):
        super(FourierPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        # Create a matrix of [position, dimension] for positional encoding
        self.position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        self.div_term = div_term

    def forward(self, positions):
        # Calculate sine and cosine positional encodings
        self.div_term = self.div_term.to(positions.device)
        pe = torch.zeros(positions.size(0), positions.size(1), self.embedding_dim, device=positions.device)
        position = positions.unsqueeze(2).float()
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe

class QFormer(nn.Module):
    def __init__(self, num_tokens, token_dim, memory_dim, num_layers, num_heads, max_memory_pos=500, use_fourier_embeddings=False):
        super(QFormer, self).__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        # Learnable tokens
        self.learnable_tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        
        # Choose positional embeddings based on configuration
        if use_fourier_embeddings:
            self.token_position_embeddings = FourierPositionalEmbedding(num_tokens, token_dim)
            self.memory_position_embeddings = FourierPositionalEmbedding(max_memory_pos, token_dim)
        else:
            self.token_position_embeddings = LearnablePositionalEmbedding(num_tokens, token_dim)
            self.memory_position_embeddings = LearnablePositionalEmbedding(max_memory_pos, token_dim)
        
        # Linear layer to project memory to the same dimension as tokens
        self.memory_projection = nn.Linear(memory_dim, token_dim)
        
        # Transformer decoder layers with batch_first=True
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=token_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
    
    def forward(self, memory, memory_mask=None):
        """
        memory: External memory input provided to the forward function, shape [batch_size, num_memory_slots, memory_dim].
        memory_mask: Optional mask for memory attention, shape [batch_size, num_memory_slots].
        """
        batch_size, actual_memory_length, _ = memory.size()
        
        # Expand learnable tokens
        tokens = self.learnable_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # invert memory mask
        if memory_mask is not None:
            memory_mask =  1.0 - memory_mask
        
        # Get positional embeddings for tokens and add them
        token_positions = torch.arange(self.num_tokens, device=memory.device).unsqueeze(0).expand(batch_size, -1)
        token_pos_embeddings = self.token_position_embeddings(token_positions)
        tokens = tokens + token_pos_embeddings
        
        # Project memory to the same dimension as tokens
        projected_memory = self.memory_projection(memory)
        
        # Get positional embeddings for memory based on its actual length and add them
        memory_positions = torch.arange(actual_memory_length, device=memory.device).unsqueeze(0).expand(batch_size, -1)
        memory_pos_embeddings = self.memory_position_embeddings(memory_positions)
        memory = projected_memory + memory_pos_embeddings

        # put memory mask to same device as memory
        if memory_mask is not None:
            memory_mask = memory_mask.to(memory.device, dtype=memory.dtype)
        
        # Process through each decoder layer
        for layer in self.decoder_layers:
            # Pass the tokens through the decoder layer with memory
            tokens = layer(tokens, memory, memory_key_padding_mask=memory_mask)
        
        return tokens
