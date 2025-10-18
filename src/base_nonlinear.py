import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from torch import einsum 
from torch import nn
from jaxtyping import Array, Float, jaxtyped
from torch import Tensor
from einops import rearrange 
from typeguard import typechecked as typechecker
from typing import List
import math

class MLP(nn.Module):
    def __init__(self, dim_d, width):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(dim_d, width)
        self.layer2 = nn.Linear(width, dim_d, bias = False)
    
    def forward(self, x):
        z = self.layer1(x)
        z = F.relu(z)
        z = self.layer2(z)
        return z
    
class CausalAttn(nn.Module):
    def __init__(self, seq_len, dim_d, dim_out, num_heads, 
        dim_r = None,
        mask = True, 
        factorize_W = True, identity_W = False, w_bias = False,
        init = 'zero', init_scale: float = 1.0,
        reparameterize_qk = False,
        reparameterize_ov = False,
        layer_norm = False,
        arc = 'softmax'):
        super(CausalAttn, self).__init__()
        self.seq_len = seq_len
        self.dim_d = dim_d

        self.dim_head = dim_d // num_heads
        self.init = init
        self.num_heads = num_heads
        self.w_q = nn.Linear(dim_d, dim_d, bias = w_bias)
        self.w_k = nn.Linear(dim_d, dim_d, bias = w_bias)
        self.w_q.weight.data.normal_(0, init_scale / math.sqrt(self.dim_head))
        # self.w_k.weight.data.normal_(0, init_scale / math.sqrt(self.dim_head))
        
        # self.w_q = nn.Parameter(
        #     torch.randn(dim_d, dim_d) * (init_scale / math.sqrt(self.dim_head))
        # )

        # self.w_k = nn.Parameter(
        #     torch.randn(dim_d, dim_d) * (init_scale / math.sqrt(self.dim_head))
        # )
        
        # self.w_v = nn.Parameter(
        #     torch.randn(dim_d, dim_d) * (init_scale / math.sqrt(self.dim_head))
        # )
        # Reparameterization trick
        if reparameterize_qk:
            self.w_k.weight.data = torch.eye(dim_d)
            self.w_k.weight.requires_grad = False
        else:
            self.w_k.weight.data.normal_(0, init_scale / math.sqrt(self.dim_head))
        
        if reparameterize_ov:
            self.w_v = nn.Linear(dim_d, dim_d, bias = w_bias)
            self.w_o = nn.Linear(dim_d, dim_out, bias = w_bias)
            self.w_v.weight.data = torch.eye(dim_d)
            self.w_v.weight.requires_grad = False
        else:
            if dim_r is None:
                dim_r = dim_d
            self.w_v = nn.Linear(dim_d, dim_r, bias = w_bias)
            self.w_o = nn.Linear(dim_r, dim_out, bias = w_bias)
            self.w_v.weight.data.normal_(0, init_scale / math.sqrt(self.dim_d))
        self.w_o.weight.data.normal_(0, init_scale / math.sqrt(self.dim_head))
        # if not reparameterize:
        #     self.w_k = nn.Linear(dim_d, dim_d, bias = w_bias)
        #     self.w_k.weight.data.normal_(0, init_scale / math.sqrt(self.dim_head))
        #     self.w_o = nn.Linear(dim_d, dim_d, bias = w_bias)
        #     self.w_o.weight.data.normal_(0, init_scale / math.sqrt(self.dim_d))
        #     # self.w_o = nn.Parameter(
        #     #     torch.randn(dim_d, dim_d) * (init_scale / math.sqrt(self.dim_d))
        #     # )
        # else:
        #     self.w_k = torch.eye(dim_d)
        #     self.w_o = torch.eye(dim_d)

        if self.init == 'zero':
            self.w_q.weight.data.zero_()
            if not reparameterize_qk:
                self.w_k.weight.data.zero_()
        elif self.init == 'identity':
            self.w_q.weight.data = 1e-4 * torch.ones_like(self.w_q.weight)
            self.w_o.weight.data = 1e-4 * torch.ones_like(self.w_o.weight)
            if not reparameterize_qk:
                self.w_k.weight.data = 1e-4 * torch.eye(self.w_k.weight.shape[0])
            if not reparameterize_ov:
                if dim_r == dim_d:
                    self.w_v.weight.data = 1e-4 * torch.eye(self.w_v.weight.shape[0])
                else:
                    self.w_v.weight.data = 1e-4 * torch.ones_like(self.w_v.weight)
        if mask:
            self.mask = torch.tril(torch.ones(seq_len, seq_len), diagonal = 0)
        else:
            self.mask = torch.eye(seq_len)
        
        self.arc = arc
        self.layer_norm = layer_norm

        # inspection
        self.attn_map = []
        
    # def _init_weights(self):
    #     for module in self.modules():
    #         import pdb; pdb.set_trace()
    #         if self.init == 'zero':
    #             if isinstance(module, nn.Linear):
    #                 # print(f"Initializing {module} to zero")
    #                 module.weight.data.zero_()
    #                 if module.bias is not None:
    #                     module.bias.data.zero_()
    #             elif isinstance(module, nn.Parameter):
    #                 module.data.zero_()
    #         elif self.init == 'default':
    #             # already initialized
    #             return 

    def forward(self, 
        X: Float[Tensor, "btz seq_len dim_d"]
    ) -> Float[Tensor, "btz seq_len dim_d"]:
        output = []
        self.attn_map = []
        Xq = rearrange(self.w_q(X), 'b t (h d) -> b h t d', h = self.num_heads)
        Xk = rearrange(self.w_k(X), 'b t (h d) -> b h t d', h = self.num_heads)
        Xv = rearrange(self.w_v(X), 'b t (h d) -> b h t d', h = self.num_heads)

        dot_prod = einsum('b h q d, b h t d -> b h q t', Xq, Xk)
        if self.arc == 'softmax':
            dot_prod = dot_prod.masked_fill(self.mask.to(dot_prod.device) == 0, float('-inf'))

            # attn = F.softmax(dot_prod, dim = -1)
            attn = F.softmax(dot_prod / math.sqrt(self.dim_head), dim = -1)
        elif self.arc == 'linear':
            dot_prod = dot_prod.masked_fill(self.mask.to(dot_prod.device) == 0, 0)
            # set dot_prod to be non-negative
            # dot_prod = F.relu(dot_prod)
            # dot_prod = dot_prod.masked_fill(dot_prod < 0, 0)
            attn = dot_prod
        else:
            raise ValueError("Invalid attention arc")
        for i in range(self.num_heads):
            self.attn_map.append(attn[:, i, :, :])
        out = einsum('b h q t, b h t d -> b h q d', attn, Xv)
        out = rearrange(out, 'b h q d -> b q (h d)')
        # if self.layer_norm:
        #     out = out / torch.norm(out, dim = -1, keepdim = True)
        out = self.w_o(out)
        return out


class TF(nn.Module):
    def __init__(self, seq_len, dim_d, vocab_size, 
                heads: List[int],
                causal = True, n_layer = 2,
                identity_W = False, factorize_W = False, identity_wout = False,
                init = 'zero',
                # attention
                reparameterize_qk = False,
                reparameterize_ov = False,
                dim_r = None,
                attn_arc = 'softmax', # ['softmax', 'linear']
                # embedding 
                wte_type = 'ortho', use_pos = False, weight_tying = True,
                # MLP
                use_mlp = False,
                mlp_width = 128,
                device = 'cuda',

                residual = False,
                layer_norm = False,
                ):
        super(TF, self).__init__() 
        assert len(heads) == n_layer
        self.n_layer = n_layer
        
        self.vocab_size = vocab_size
        if wte_type == 'ortho':
            # round to the smallest power of 2
            # dim_d = 2 ** math.ceil(math.log2(vocab_size))
            self.wte = torch.eye(dim_d)
        else:
            dim_d = 2 ** math.ceil(math.log2(dim_d))
            self.wte = torch.randn(vocab_size, dim_d)
            self.wte = self.wte / torch.norm(self.wte, dim = -1, keepdim = True)
        self.wte = nn.Parameter(self.wte, requires_grad = False)

        self.dim_d = dim_d
        self_attns = []
        for i in range(n_layer):
            self_attns.append(
                CausalAttn(seq_len, dim_d, dim_d, heads[i], 
                dim_r = dim_r,
                mask = causal, 
                factorize_W = factorize_W, identity_W = identity_W, init = init, 
                reparameterize_qk = reparameterize_qk, 
                reparameterize_ov = reparameterize_ov,
                layer_norm = layer_norm,
                arc = attn_arc) 
            )
        self.self_attn = nn.ModuleList(self_attns)
        if use_mlp:
            mlp_width = 2 * dim_d if mlp_width is None else mlp_width
            self.mlp = nn.ModuleList([MLP(dim_d, mlp_width) for _ in range(n_layer)])
        self.use_mlp = use_mlp
        if wte_type == 'ortho' or weight_tying:
            self.unembed = self.wte
        else:
            self.unembed = torch.randn(vocab_size, dim_d)
            self.unembed = self.unembed / torch.norm(self.unembed, dim = -1, keepdim = True)
        self.unembed = nn.Parameter(self.unembed, requires_grad = False)

        self.init = init
        self.residual = residual

    def embed(self, 
            input_seq: Float[Tensor, "btz seq_len"],
            # position embedding
            pos_embed = None, 
        ):
        # if use_one_hot:
        #     token_embed = F.one_hot(input_seq, self.vocab_size).float()
        # else:
        #     token_embed = torch.randn
        token_embed = self.wte[input_seq]
        # import pdb; pdb.set_trace() # test if the shape is correct

        # if pos_embed is None:
        #     pos_embed = torch.eye(input_seq.shape[-1]).to(input_seq.device)
        #     pos_embed = pos_embed.expand(*input_seq.shape, input_seq.shape[-1])
        # else:
        #     assert pos_embed.shape == (*input_seq.shape, input_seq.shape[-1])
        # # [n, S, d_0 = S + K] 
        # if use_pos:
        #     return torch.cat([token_embed, pos_embed], dim=-1)
        # else:
        return token_embed

    def forward(self, 
        input_seq: Float[Tensor, "btz seq_len"],
        pos = None
    ) -> Float[Tensor, "btz vocab_size"]:
        # input_seq: [n, S]
        # pos: [n, S, K]
        # output: [n, K] 
        x = self.embed(input_seq, pos)
        for i in range(self.n_layer):
            attn = self.self_attn[i](x)
            # print("attn shape:", attn.shape)
            # print("x shape:", x.shape)
            if self.use_mlp:
                mlp = self.mlp[i](attn)
                if self.residual:
                    x = x + mlp + attn
                else:
                    x = mlp + attn
            else:
                # TODO: include the input?
                if self.residual and i < self.n_layer - 1:
                # if self.residual:
                    x = x + attn
                else: 
                    x = attn   
            # if i == 0 and self.layer_norm:
            #     x = x / torch.norm(x, dim = -1, keepdim = True)
                
        x = x[..., -1, :] # Only use the last token for query
        return F.softmax(x @ self.unembed.T, dim = -1)

class DisentangledTF(nn.Module):
    def __init__(self, seq_len, dim_d, vocab_size, 
                heads: List[int],
                causal = True, n_layer = 2,
                identity_W = False, factorize_W = False, identity_wout = False,
                init = 'zero',
                # attention
                reparameterize_qk = False,
                reparameterize_ov = False,
                dim_r = None,
                attn_arc = 'softmax', # ['softmax', 'linear']
                # embedding 
                wte_type = 'ortho', use_pos = False, weight_tying = True,
                # MLP
                use_mlp = False,
                mlp_width = 128,
                device = 'cuda',

                residual = False,
                layer_norm = False,
                ):
        super(DisentangledTF, self).__init__() 
        assert len(heads) == n_layer
        self.n_layer = n_layer
        
        self.vocab_size = vocab_size
        if wte_type == 'ortho':
            # round to the smallest power of 2
            dim_d = 2 ** math.ceil(math.log2(vocab_size))
            self.wte = torch.eye(dim_d)
        else:
            dim_d = 2 ** math.ceil(math.log2(dim_d))
            self.wte = torch.randn(vocab_size, dim_d)
            self.wte = self.wte / torch.norm(self.wte, dim = -1, keepdim = True)
        self.wte = nn.Parameter(self.wte, requires_grad = False)

        self.dim_d = dim_d
        self_attns = []
        dim_di = dim_d
        for i in range(n_layer):
            self_attns.append(
                CausalAttn(seq_len, dim_di, dim_d, heads[i], 
                dim_r = dim_r,
                mask = causal, 
                factorize_W = factorize_W, identity_W = identity_W, init = init, 
                reparameterize_qk = reparameterize_qk, 
                reparameterize_ov = reparameterize_ov,
                layer_norm = layer_norm,
                arc = attn_arc) 
            )
            if residual:
                dim_di = dim_d * 2
        self.self_attn = nn.ModuleList(self_attns)
        if use_mlp:
            mlp_width = 2 * dim_d if mlp_width is None else mlp_width
            self.mlp = nn.ModuleList([MLP(dim_d, mlp_width) for _ in range(n_layer)])
        self.use_mlp = use_mlp
        if wte_type == 'ortho' or weight_tying:
            self.unembed = self.wte
        else:
            self.unembed = torch.randn(vocab_size, dim_d)
            self.unembed = self.unembed / torch.norm(self.unembed, dim = -1, keepdim = True)
        self.unembed = nn.Parameter(self.unembed, requires_grad = False)

        self.init = init
        self.residual = residual

    def embed(self, 
            input_seq: Float[Tensor, "btz seq_len"],
            # position embedding
            pos_embed = None, 
        ):
        # if use_one_hot:
        #     token_embed = F.one_hot(input_seq, self.vocab_size).float()
        # else:
        #     token_embed = torch.randn
        token_embed = self.wte[input_seq]
        # import pdb; pdb.set_trace() # test if the shape is correct

        # if pos_embed is None:
        #     pos_embed = torch.eye(input_seq.shape[-1]).to(input_seq.device)
        #     pos_embed = pos_embed.expand(*input_seq.shape, input_seq.shape[-1])
        # else:
        #     assert pos_embed.shape == (*input_seq.shape, input_seq.shape[-1])
        # # [n, S, d_0 = S + K] 
        # if use_pos:
        #     return torch.cat([token_embed, pos_embed], dim=-1)
        # else:
        return token_embed

    def forward(self, 
        input_seq: Float[Tensor, "btz seq_len"],
        pos = None
    ) -> Float[Tensor, "btz vocab_size"]:
        # input_seq: [n, S]
        # pos: [n, S, K]
        # output: [n, K] 
        x = self.embed(input_seq, pos)
        for i in range(self.n_layer):
            attn = self.self_attn[i](x)
            # print("attn shape:", attn.shape)
            # print("x shape:", x.shape)
            if self.use_mlp:
                mlp = self.mlp[i](attn)
                if self.residual:
                    x = x + mlp + attn
                else:
                    x = mlp + attn
            else:
                # TODO: include the input?
                if self.residual and i < self.n_layer - 1:
                # if self.residual:
                    x = torch.cat([x, attn], dim = -1)
                else: 
                    x = attn   
            # if i == 0 and self.layer_norm:
            #     x = x / torch.norm(x, dim = -1, keepdim = True)
                
        x = x[..., -1, :] # Only use the last token for query
        return F.softmax(x @ self.unembed.T, dim = -1)
