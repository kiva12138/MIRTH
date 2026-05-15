import torch.nn as nn
from config.config_vla import ACTION_DIM, NUM_ACTIONS_CHUNK
import math
import torch


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_token_type="one_for_action_chunk", # options: "one_for_action_chunk", "one_for_action_step", "one_for_action_dim"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_token_type = action_token_type
        
        if self.action_token_type == "one_for_action_chunk":
            self.action_token_num = 1
            self.model = MLPResNet(num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=ACTION_DIM*NUM_ACTIONS_CHUNK)
        elif self.action_token_type == "one_for_action_step":
            self.action_token_num = NUM_ACTIONS_CHUNK
            self.model = MLPResNet(num_blocks=2, input_dim=input_dim*NUM_ACTIONS_CHUNK, hidden_dim=hidden_dim, output_dim=ACTION_DIM*NUM_ACTIONS_CHUNK) 
        elif self.action_token_type == "one_for_action_dim":
            self.action_token_num = NUM_ACTIONS_CHUNK * ACTION_DIM
            self.model = MLPResNet(num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=ACTION_DIM)
        else:
            raise ValueError(f"Invalid action_token_type: {action_token_type}")
        
    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, self.action_token_num, input_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, NUM_ACTIONS_CHUNK, action_dim)
        batch_size = actions_hidden_states.shape[0]
        
        if self.action_token_type == "one_for_action_chunk":
            action = self.model(actions_hidden_states.reshape(batch_size, self.input_dim))
            action = action.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM)  # shape: (batch_size, NUM_ACTIONS_CHUNK, action_dim)
        elif self.action_token_type == "one_for_action_step":
            action = self.model(actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK * self.input_dim))
            action = action.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM)  # shape: (batch_size, NUM_ACTIONS_CHUNK, action_dim)
        elif self.action_token_type == "one_for_action_dim":
             rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM * self.input_dim) 
             action = self.model(rearranged_actions_hidden_states)
        else:
            raise ValueError(f"Invalid action_token_type: {self.action_token_type}")
        
        return action