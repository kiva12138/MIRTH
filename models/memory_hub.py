'''
The core memory hub operation for MIRTH.
In each memory hub, we have two types of EMA, one is run_ema_for_training and the other is run_ema_for_training_with_loop.
Theoretically, the vectorized version (run_ema_for_training) should be faster. But in practice, we found the loop version is faster for now. So we keep both versions here for record and future optimization. In practical training, we use the loop version.
'''

import math
import torch
import torch.nn as nn
from collections import deque


class VisionMemoryHubForTraining(nn.Module):
    def __init__(self, hidden_dim, long_memory_scale_number=4, short_memory_length=4, tau=8.0, beta_min=0.01, beta_max=0.3, gamma=0.2, lmbd=0.2, bias=1.0, prefix_type="union"):
        super().__init__()
        # For long term memory with multiple decay rates
        self.long_memory_scale_number = long_memory_scale_number
        self.betas = torch.exp(torch.linspace(math.log(beta_min), math.log(beta_max), self.long_memory_scale_number))
        self.Mbank = None   # list of tensors: [B,N,D] * long_memory_scale_number
        self.gamma = gamma  # speed EMA
        self.lmbd  = lmbd   # variant EMA
        self.V = None       # [B,N,D]
        self.S = None       # [B,N,D]
        self.Uv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Us = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wa = nn.Linear(3*hidden_dim, long_memory_scale_number)      # Produce alpha_t
        
        # For short term memory queue
        self.layer_norm_queue = nn.LayerNorm(hidden_dim)
        self.short_memory_length = short_memory_length
        self.queue = None   # [B,w,N,D]
        self.tau = tau
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = bias

        # For final fusion of long and short term memories
        self.prefix_type = prefix_type
        if self.prefix_type == "union":
            self.Wg = nn.Linear(3*hidden_dim, hidden_dim)      # Optional, fuse long-short term memory
        else:
            pass


    def run_vision_ema_for_training(self, datas, process_module, pad_mask): 
        # Datas include the history frames (without current frame)
        # Datas: {'dino': torch.ones(B, num_cameras, 50, C, H, W), 'siglip': torch.ones(B, num_cameras, 50, C, H, W)}

        # If the cameras are moved, we should align frames here. A general implementation would be simply cross-attention.
        # For simplicity, we skip this step in this implementation.
        
        B, num_camera, history_len, C, H, W = datas['dino'].shape
        datas = {
            key: data.permute(0, 2, 1, 3, 4, 5).reshape(B * history_len, num_camera, C, H, W) 
            for key, data in datas.items()
        } # each with shape (B*(history_len+1), num_cameras, C, H, W)

        patch_embeddings = process_module(datas)  # (B*(history_len+1), num_cameras * num_patches, D)
        patch_embeddings = patch_embeddings.unflatten(0, (B, history_len))  # (B, (history_len+1), num_cameras * num_patches, D)
        device, dtype = patch_embeddings.device, patch_embeddings.dtype
        # patch_embeddings[~pad_mask.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, patch_embeddings.shape[2], patch_embeddings.shape[3]))] = 0.0
        patch_embeddings[~pad_mask] = 0.0
        most_recent_patch_embeddings = patch_embeddings[:, -1, :, :]  # (B, num_cameras * num_patches, D)

        num_patches, dimension = patch_embeddings.shape[2], patch_embeddings.shape[3]
        T = history_len
        L = max(T - 1, 0)  # exclude the last frame for Mbank and queue

        # Vectorized computation of long- and short-term EMA statistics over time
        # Compute deltas against previous frame (queue[-1] semantics with initial zeros)
        prev = torch.cat([
            torch.zeros(B, 1, num_patches, dimension, device=device, dtype=dtype),
            patch_embeddings[:, :-1]
        ], dim=1)  # (B,T,N,D)
        delta = patch_embeddings - prev  # (B,T,N,D)

        # Long-term EMA for each beta using only first L frames
        betas = self.betas.to(device=device, dtype=dtype)  # (M,)
        if L > 0:
            t_idx = torch.arange(L, device=device, dtype=dtype)  # (L,)
            decay_powers_m = (L - 1 - t_idx)[None, :]  # (1,L)
            weights_M = betas[:, None] * torch.pow(1 - betas[:, None], decay_powers_m)  # (M,L) where L corresponds to time index t
            # Correct einsum: match time dimension between patch embeddings (t) and weights (t) so we form Σ_t x_t * w_{m,t}
            M_stack = torch.einsum('btnd,mt->bndm', patch_embeddings[:, :L], weights_M)  # (B,N,D,M)
            self.Mbank = [M_stack[..., m] for m in range(self.long_memory_scale_number)]
        else:
            # No history available -> zeros
            self.Mbank = [torch.zeros(B, num_patches, dimension, device=device, dtype=dtype) for _ in range(self.long_memory_scale_number)]

        # Velocity and variance EMAs over deltas across all T frames
        gamma = torch.as_tensor(self.gamma, device=device, dtype=dtype)
        lmbd = torch.as_tensor(self.lmbd, device=device, dtype=dtype)
        t_idx_full = torch.arange(T, device=device, dtype=dtype)
        decay_powers_full = (T - 1 - t_idx_full)
        weights_V = gamma * torch.pow(1 - gamma, decay_powers_full)  # (T,)
        weights_S = lmbd * torch.pow(1 - lmbd, decay_powers_full)    # (T,)
        self.V = torch.einsum('btnd,t->bnd', delta, weights_V)             # (B,N,D)
        self.S = torch.einsum('btnd,t->bnd', delta*delta, weights_S)       # (B,N,D)

        # Build the final short-term queue content (oldest to newest), using only first L frames
        self.queue = deque(maxlen=self.short_memory_length)
        if L >= self.short_memory_length:
            recent = patch_embeddings[:, L - self.short_memory_length:L]  # (B,w,N,D)
        else:
            pad = torch.zeros(B, self.short_memory_length - L, num_patches, dimension, device=device, dtype=dtype)
            recent = torch.cat([pad, patch_embeddings[:, :L]], dim=1)     # (B,w,N,D)
        for i in range(self.short_memory_length):
            self.queue.append(recent[:, i])

        return self.Mbank, self.V, self.S, self.queue, most_recent_patch_embeddings


    def run_vision_ema_for_training_with_loop(self, datas, process_module, pad_mask): 
        # Datas include the history frames (with current frame)
        # Datas: {'dino': torch.ones(B, num_cameras, 50, C, H, W), 'siglip': torch.ones(B, num_cameras, 50, C, H, W)}
        # Pad mask: (B, 51)
        
        B, num_camera, history_len, C, H, W = datas['dino'].shape
        datas = {
            key: data.permute(0, 2, 1, 3, 4, 5).reshape(B * history_len, num_camera, C, H, W) 
            for key, data in datas.items()
        } # each with shape (B*(history_len+1), num_cameras, C, H, W)

        patch_embeddings = process_module(datas)  # (B*(history_len+1), num_cameras * num_patches, D)
        patch_embeddings = patch_embeddings.unflatten(0, (B, history_len))  # (B, (history_len+1), num_cameras * num_patches, D)
        device, dtype = patch_embeddings.device, patch_embeddings.dtype
        # patch_embeddings[~pad_mask.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, patch_embeddings.shape[2], patch_embeddings.shape[3]))] = 0.0
        patch_embeddings[~pad_mask] = 0.0
        most_recent_patch_embeddings = patch_embeddings[:, -1, :, :]  # (B, num_cameras * num_patches, D)

        num_patches, dimension = patch_embeddings.shape[2], patch_embeddings.shape[3]
        
        self.Mbank = [torch.zeros(B, num_patches, dimension, device=device, dtype=dtype) for _ in range(self.long_memory_scale_number)]
        self.V = torch.zeros(B, num_patches, dimension, device=device, dtype=dtype)
        self.S = torch.zeros(B, num_patches, dimension, device=device, dtype=dtype)
        self.queue = deque(maxlen=self.short_memory_length)
        for _ in range(self.short_memory_length):
            self.queue.append(torch.zeros(B, num_patches, dimension, device=device, dtype=dtype))

        for t in range(history_len):
            delta = patch_embeddings[:, t] - self.queue[-1]
            
            if t < history_len - 1:
                new_Mbank = []
                for m, beta in enumerate(self.betas):
                    new_M = (1-beta) * self.Mbank[m] + beta * patch_embeddings[:, t]
                    new_Mbank.append(new_M)
                self.Mbank = new_Mbank
                self.queue.append(patch_embeddings[:, t])
            else:
                pass
            
            self.V = (1-self.gamma) * self.V + self.gamma * delta
            self.S = (1-self.lmbd) * self.S + self.lmbd * (delta*delta)

        return self.Mbank, self.V, self.S, self.queue, most_recent_patch_embeddings


    def forward(self, pixel_values, pixel_values_history, vison_backbone, pad_mask):
        # Run vectorized EMA accumulation over history frames
        # Theoretically run_vision_ema_for_training should be faster, but in practice the loop version is faster for now. 
        # Mbank, V, S, recent_queue = self.run_vision_ema_for_training(pixel_values, vison_backbone)
        pixel_values_history_ = {}
        with torch.no_grad():
            pixel_values_history_['dino'] = torch.cat([pixel_values_history['dino'], pixel_values['dino'].unsqueeze(2)], dim=2)
            pixel_values_history_['siglip'] = torch.cat([pixel_values_history['siglip'], pixel_values['siglip'].unsqueeze(2)], dim=2)
            Mbank, V, S, recent_queue, final_x_t_ = self.run_vision_ema_for_training_with_loop(pixel_values_history_, vison_backbone, pad_mask)

        feat_a = torch.cat([final_x_t_, V, torch.sqrt(S + 1e-6)], dim=-1)    # (B,N,3D)
        alpha = torch.softmax(self.Wa(feat_a), dim=-1).unsqueeze(-2)         # (B,N,1,M)
        M_stack = torch.stack(Mbank, dim=-1)                                 # (B,N,D,M)
        W_t = (M_stack * alpha).sum(-1)                                      # (B,N,D)
        W_t = W_t + self.Uv(V) + self.Us(torch.sqrt(S + 1e-6))               # The final long term memory output

        # For short term memory update
        queue_normed = self.layer_norm_queue(torch.stack(list(recent_queue), dim=1))  # (B,w,N,D)      
        k = self.Wk(queue_normed)                                                     # (B,w,N,D)
        q = self.Wq(final_x_t_).unsqueeze(1)                                          # (B,1,N,D)
        sim = (q * k).sum(-1) * self.tau                                              # (B,w,N) Dot product attention
        j_idx = torch.arange(self.short_memory_length, device=final_x_t_.device)[None, :, None]  # (1,w,1)
        sim = sim + self.bias * j_idx                                     # Larger bias means more attention to recent frames
        pi = torch.softmax(sim, dim=1).unsqueeze(-1)                      # (B,w,N,1)
        v = self.Wv(queue_normed)                                         # (B,w,N,D)
        S_short = (pi * v).sum(1)                                         # (B,N,D)

        # Final memory fusion
        if self.prefix_type == "union":
            gate = torch.sigmoid(self.Wg(torch.cat([final_x_t_, W_t, S_short], dim=-1)))  # (B,N,D)
            Mem_t = gate * W_t + (1-gate) * S_short
        else:
            Mem_t = None
        return {"workspace": W_t, "short": S_short, "mem": Mem_t}


class ProprioMemoryHubForTraining(nn.Module):
    def __init__(self, hidden_dim, long_memory_scale_number=4, short_memory_length=4, tau=8.0, beta_min=0.01, beta_max=0.3, gamma=0.2, lmbd=0.2, bias=1.0, prefix_type="union"):
        super().__init__()
        # For long term memory with multiple decay rates
        self.long_memory_scale_number = long_memory_scale_number
        self.betas = torch.exp(torch.linspace(math.log(beta_min), math.log(beta_max), self.long_memory_scale_number))
        self.Mbank = None   # list of tensors: [B,N,D] * long_memory_scale_number
        self.gamma = gamma  # speed EMA
        self.lmbd  = lmbd   # variant EMA
        self.V = None       # [B,N,D]
        self.S = None       # [B,N,D]
        self.Uv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Us = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wa = nn.Linear(3*hidden_dim, long_memory_scale_number)      # Produce alpha_t
        
        # For short term memory queue
        self.layer_norm_queue = nn.LayerNorm(hidden_dim)
        self.short_memory_length = short_memory_length
        self.queue = None   # [B,w,N,D]
        self.tau = tau
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = bias

        # For final fusion of long and short term memories
        self.prefix_type = prefix_type
        if self.prefix_type == "union":
            self.Wg = nn.Linear(3*hidden_dim, hidden_dim)      # Optional, fuse long-short term memory
        else:
            pass


    def run_proprio_ema_for_training(self, datas, pad_mask): 
        # Vectorized EMA that exactly matches the loop semantics in
        # `run_proprio_ema_for_training_with_loop`.
        # Inputs:
        # - datas: (B, T, D) sequence including the current frame at the end
        # - pad_mask: (B, T) where True indicates valid timesteps
        # Outputs mirror the loop version: Mbank (list of BxD), V (BxD), S (BxD),
        #   queue (deque of length w with BxD tensors for the first T-1 frames),
        #   and most_recent_embeddings (BxD, the current frame embedding).

        B, T, D = datas.shape
        embeddings = datas  # (B, T, D)
        device, dtype = embeddings.device, embeddings.dtype

        # Apply padding to match loop behavior (invalid steps become zeros)
        mask = pad_mask.unsqueeze(-1).to(dtype=torch.bool)  # (B,T,1)
        embeddings = embeddings.masked_fill(~mask, 0.0)
        most_recent_embeddings = embeddings[:, -1, :]  # (B, D)

        # Exclude the last (current) frame for Mbank and short-term queue
        L = max(T - 1, 0)

        # Delta_t uses previous frame with initial zero (queue[-1] == 0 at t=0)
        prev = torch.cat(
            [torch.zeros(B, 1, D, device=device, dtype=dtype), embeddings[:, :-1]], dim=1
        )  # (B,T,D)
        delta = embeddings - prev  # (B,T,D)

        # Long-term EMA over the first L frames (exclude current)
        betas = self.betas.to(device=device, dtype=dtype)  # (M,)
        if L > 0:
            t_idx = torch.arange(L, device=device, dtype=dtype)  # (L,)
            decay = (L - 1 - t_idx)[None, :]  # (1,L)
            weights_M = betas[:, None] * torch.pow(1 - betas[:, None], decay)  # (M,L)
            # Sum over time dimension to match iterative EMA: Σ_t x_t * beta*(1-beta)^(L-1-t)
            M_stack = torch.einsum('btd,mt->bdm', embeddings[:, :L], weights_M)  # (B,D,M)
            self.Mbank = [M_stack[..., m] for m in range(self.long_memory_scale_number)]
        else:
            self.Mbank = [torch.zeros(B, D, device=device, dtype=dtype) for _ in range(self.long_memory_scale_number)]

        # Velocity and variance EMA over all T deltas
        gamma = torch.as_tensor(self.gamma, device=device, dtype=dtype)
        lmbd = torch.as_tensor(self.lmbd, device=device, dtype=dtype)
        t_idx_full = torch.arange(T, device=device, dtype=dtype)
        decay_full = (T - 1 - t_idx_full)
        w_V = gamma * torch.pow(1 - gamma, decay_full)  # (T,)
        w_S = lmbd * torch.pow(1 - lmbd, decay_full)    # (T,)
        self.V = torch.einsum('btd,t->bd', delta, w_V)              # (B,D)
        self.S = torch.einsum('btd,t->bd', delta * delta, w_S)      # (B,D)

        # Short-term queue composed of the first L frames (padded on the left with zeros if needed)
        self.queue = deque(maxlen=self.short_memory_length)
        if L >= self.short_memory_length:
            recent = embeddings[:, L - self.short_memory_length:L]  # (B,w,D)
        else:
            pad_needed = self.short_memory_length - L
            pad_tensor = torch.zeros(B, pad_needed, D, device=device, dtype=dtype)
            recent = torch.cat([pad_tensor, embeddings[:, :L]], dim=1)  # (B,w,D)
        for i in range(self.short_memory_length):
            self.queue.append(recent[:, i, :])  # (B,D)

        return self.Mbank, self.V, self.S, self.queue, most_recent_embeddings


    def run_proprio_ema_for_training_with_loop(self, datas, pad_mask): 
        # Datas include the history frames (with current frame)
        # Datas: torch.ones(B, 51, D)
        # Pad mask: (B, 51)
        
        B, history_len, D = datas.shape

        embeddings = datas  # (B, history_len+1, D)
        device, dtype = embeddings.device, embeddings.dtype
        # embeddings[~pad_mask.unsqueeze(-1).expand((-1, -1, embeddings.shape[2]))] = 0.0
        embeddings[~pad_mask] = 0.0
        most_recent_embeddings = embeddings[:, -1, :]  # (B, D)

        dimension = D
        
        self.Mbank = [torch.zeros(B, dimension, device=device, dtype=dtype) for _ in range(self.long_memory_scale_number)]
        self.V = torch.zeros(B, dimension, device=device, dtype=dtype)
        self.S = torch.zeros(B, dimension, device=device, dtype=dtype)
        self.queue = deque(maxlen=self.short_memory_length)
        for _ in range(self.short_memory_length):
            self.queue.append(torch.zeros(B, dimension, device=device, dtype=dtype))

        for t in range(history_len):
            delta = embeddings[:, t] - self.queue[-1]

            if t < history_len - 1:
                new_Mbank = []
                for m, beta in enumerate(self.betas):
                    new_M = (1-beta) * self.Mbank[m] + beta * embeddings[:, t]
                    new_Mbank.append(new_M)
                self.Mbank = new_Mbank
                self.queue.append(embeddings[:, t])
            else:
                pass
            
            self.V = (1-self.gamma) * self.V + self.gamma * delta
            self.S = (1-self.lmbd) * self.S + self.lmbd * (delta*delta)

        return self.Mbank, self.V, self.S, self.queue, most_recent_embeddings


    def forward(self, proprio, proprio_history, pad_mask):
        # Run vectorized EMA accumulation over history frames
        with torch.no_grad():
            proprio_history_ = torch.cat([proprio_history, proprio.unsqueeze(1)], dim=1)
            Mbank, V, S, recent_queue, final_x_t_ = self.run_proprio_ema_for_training_with_loop(proprio_history_, pad_mask)

        feat_a = torch.cat([final_x_t_, V, torch.sqrt(S + 1e-6)], dim=-1)  # (B,3D)
        alpha = torch.softmax(self.Wa(feat_a), dim=-1).unsqueeze(-2)       # (B,1,M)
        M_stack = torch.stack(Mbank, dim=-1)                               # (B,D,M)
        W_t = (M_stack * alpha).sum(-1)                                    # (B,D)
        W_t = W_t + self.Uv(V) + self.Us(torch.sqrt(S + 1e-6))             # The final long term memory output

        # For short term memory update
        queue_normed = self.layer_norm_queue(torch.stack(list(recent_queue), dim=1))  # (B,w,D)      
        k = self.Wk(queue_normed)                                                     # (B,w,D)
        q = self.Wq(final_x_t_).unsqueeze(1)                                          # (B,1,D)
        sim = (q * k).sum(-1) * self.tau                                              # (B,w) Dot product attention
        j_idx = torch.arange(self.short_memory_length, device=final_x_t_.device)[None, :]  # (1,w)
        sim = sim + self.bias * j_idx                                     # Larger bias means more attention to recent frames
        pi = torch.softmax(sim, dim=1).unsqueeze(-1)                      # (B,w,1)
        v = self.Wv(queue_normed)                                         # (B,w,D)
        S_short = (pi * v).sum(1)                                         # (B,D)

        # Final memory fusion
        if self.prefix_type == "union":
            gate = torch.sigmoid(self.Wg(torch.cat([final_x_t_, W_t, S_short], dim=-1)))  # (B,N,D)
            Mem_t = gate * W_t + (1-gate) * S_short
        else:
            Mem_t = None
        return {"workspace": W_t, "short": S_short, "mem": Mem_t}


class ActionMemoryHubForTraining(nn.Module):
    '''
    This memory hub is designed for action memory. But we found this may cause some information leakage. So we keep it here for record but do not use it in the final version.
    '''
    def __init__(self, hidden_dim, long_memory_scale_number=4, short_memory_length=4, tau=8.0, beta_min=0.01, beta_max=0.3, gamma=0.2, lmbd=0.2, bias=1.0, prefix_type="union"):
        super().__init__()
        # For long term memory with multiple decay rates
        self.long_memory_scale_number = long_memory_scale_number
        self.betas = torch.exp(torch.linspace(math.log(beta_min), math.log(beta_max), self.long_memory_scale_number))
        self.Mbank = None   # list of tensors: [B,N,D] * long_memory_scale_number
        self.gamma = gamma  # speed EMA
        self.lmbd  = lmbd   # variant EMA
        self.V = None       # [B,N,D]
        self.S = None       # [B,N,D]
        self.Uv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Us = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wa = nn.Linear(3*hidden_dim, long_memory_scale_number)      # Produce alpha_t
        
        # For short term memory queue
        self.layer_norm_queue = nn.LayerNorm(hidden_dim)
        self.short_memory_length = short_memory_length
        self.queue = None   # [B,w,N,D]
        self.tau = tau
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = bias

        # For final fusion of long and short term memories
        self.prefix_type = prefix_type
        if self.prefix_type == "union":
            self.Wg = nn.Linear(3*hidden_dim, hidden_dim)      # Optional, fuse long-short term memory
        else:
            pass


    def run_action_ema_for_training_with_loop(self, datas, pad_mask): 
        # Datas include the history frames (with current frame)
        # Datas: torch.ones(B, 51, D)
        # Pad mask: (B, 51)
        
        B, history_len, D = datas.shape

        embeddings = datas  # (B, history_len+1, D)
        device, dtype = embeddings.device, embeddings.dtype
        # embeddings[~pad_mask[:, :history_len].unsqueeze(-1).expand((-1, -1, embeddings.shape[2]))] = 0.0
        embeddings[~pad_mask[:, :history_len]] = 0.0
        most_recent_embeddings = embeddings[:, -1, :]  # (B, D)

        dimension = D
        
        self.Mbank = [torch.zeros(B, dimension, device=device, dtype=dtype) for _ in range(self.long_memory_scale_number)]
        self.V = torch.zeros(B, dimension, device=device, dtype=dtype)
        self.S = torch.zeros(B, dimension, device=device, dtype=dtype)
        self.queue = deque(maxlen=self.short_memory_length)
        for _ in range(self.short_memory_length):
            self.queue.append(torch.zeros(B, dimension, device=device, dtype=dtype))

        for t in range(history_len):
            delta = embeddings[:, t] - self.queue[-1]

            new_Mbank = []
            for m, beta in enumerate(self.betas):
                new_M = (1-beta) * self.Mbank[m] + beta * embeddings[:, t]
                new_Mbank.append(new_M)
            self.Mbank = new_Mbank
            self.queue.append(embeddings[:, t])
            
            self.V = (1-self.gamma) * self.V + self.gamma * delta
            self.S = (1-self.lmbd) * self.S + self.lmbd * (delta*delta)

        return self.Mbank, self.V, self.S, self.queue, most_recent_embeddings


    def forward(self, action, action_history, pad_mask):
        # Run vectorized EMA accumulation over history frames
        with torch.no_grad():
            # action_history_ = torch.cat([action_history, action.unsqueeze(1)], dim=1)
            action_history_ = action_history
            Mbank, V, S, recent_queue, final_x_t_ = self.run_action_ema_for_training_with_loop(action_history_, pad_mask)

        feat_a = torch.cat([final_x_t_, V, torch.sqrt(S + 1e-6)], dim=-1)  # (B,3D)
        alpha = torch.softmax(self.Wa(feat_a), dim=-1).unsqueeze(-2)       # (B,1,M)
        M_stack = torch.stack(Mbank, dim=-1)                               # (B,D,M)
        W_t = (M_stack * alpha).sum(-1)                                    # (B,D)
        W_t = W_t + self.Uv(V) + self.Us(torch.sqrt(S + 1e-6))             # The final long term memory output

        # For short term memory update
        queue_normed = self.layer_norm_queue(torch.stack(list(recent_queue), dim=1))  # (B,w,D)      
        k = self.Wk(queue_normed)                                                     # (B,w,D)
        q = self.Wq(final_x_t_).unsqueeze(1)                                          # (B,1,D)
        sim = (q * k).sum(-1) * self.tau                                              # (B,w) Dot product attention
        j_idx = torch.arange(self.short_memory_length, device=final_x_t_.device)[None, :]  # (1,w)
        sim = sim + self.bias * j_idx                                     # Larger bias means more attention to recent frames
        pi = torch.softmax(sim, dim=1).unsqueeze(-1)                      # (B,w,1)
        v = self.Wv(queue_normed)                                         # (B,w,D)
        S_short = (pi * v).sum(1)                                         # (B,D)

        # Final memory fusion
        if self.prefix_type == "union":
            gate = torch.sigmoid(self.Wg(torch.cat([final_x_t_, W_t, S_short], dim=-1)))  # (B,N,D)
            Mem_t = gate * W_t + (1-gate) * S_short
        else:
            Mem_t = None
        return {"workspace": W_t, "short": S_short, "mem": Mem_t}


