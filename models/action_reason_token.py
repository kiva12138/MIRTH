import torch
import torch.nn as nn

class PromptEncoder(nn.Module):
    '''
    Initialze a bunch of learnable reasoning tokens from current obseractions.
    '''
    
    def __init__(self, d_model, m=8, hidden=512, p_drop=0.0, out_scale=1.0):   # m: 8, 16, 32, ... hidden: 512, 1024, 4096, low rank: 182, 64, 32
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, hidden)
        self.act = nn.SiLU()
        self.proj2 = nn.Linear(hidden, m * d_model, bias=False)
        nn.init.normal_(self.proj1.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj2.weight, mean=0.0, std=1e-3)
        
        self.dropout = nn.Dropout(p_drop)
        self.m, self.d = m, d_model
        self.register_buffer("scale", torch.tensor(out_scale))
        
    def forward(self, L_embeds, L_mask):             # [B, nL, d]
        L_mask = L_mask.unsqueeze(-1).to(L_mask.dtype)
        L_embeds = L_embeds * L_mask
        mask_counts = L_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        L_embeds = L_embeds.sum(dim=1, keepdim=True) / mask_counts
        pooled = L_embeds.squeeze(1)  # (B, llm_dim)
        pooled = self.norm(pooled)
        
        P = self.proj2(self.act(self.proj1(pooled)))  # [B, m*d]
        
        P = P.view(-1, self.m, self.d)  # [B, m, d]
        P = self.dropout(P) * self.scale
        
        return P


if __name__ == "__main__":
    B, nL, d = 2, 8, 4096
    L_embeds = torch.randn(B, nL, d)
    f_theta = PromptEncoder(d_model=4096, m=4, hidden=4096).train()
    P = f_theta(L_embeds, torch.ones(B, nL).to(torch.bool))
    print(P.shape)
