# -*- coding: utf-8 -*-

#Improved Hybrid SVD-LoRA Layer
class SVD_LoRA_Linear(nn.Module):
    def __init__(self, orig: nn.Linear, compression_rank: int, lora_rank: int, dtype=torch.bfloat16):
        super().__init__()

        # --- Part 1: The Frozen, Compressed SVD Components ---
        W_fp32 = orig.weight.data.to(torch.float32)
        b_fp32 = orig.bias.data.to(torch.float32) if orig.bias is not None else None

        U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
        U_k, S_k, Vh_k = U[:, :compression_rank], S[:compression_rank], Vh[:compression_rank, :]
        SVh_fp32 = torch.diag(S_k) @ Vh_k

        self.register_buffer('U', U_k.to(dtype))
        self.register_buffer('SVh', SVh_fp32.to(dtype))
        if b_fp32 is not None: self.register_buffer('bias', b_fp32.to(dtype))
        else: self.bias = None

        # --- Part 2: The Trainable LoRA Components with Better Initialization ---
        in_features, out_features = orig.in_features, orig.out_features

        # Kaiming initialization for better convergence
        self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Scale factor for LoRA outputs
        self.scaling = 1.0 / math.sqrt(lora_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        svd_out = F.linear(x, self.SVh)
        svd_out = F.linear(svd_out, self.U, self.bias)

        lora_out = F.linear(x, self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling

        return svd_out + lora_out
