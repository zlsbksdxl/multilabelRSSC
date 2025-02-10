import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
# from mamba_ssm import Mamba
import selective_scan_cuda
from causal_conv1d import causal_conv1d_fn
import torch.nn.functional as F

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
# pytorch cross scan =============

def compute_attn_matrix_fn(delta, delta_bias, A, B, C, L, x_shape, dtype=torch.float16):
    dt = F.softplus(delta + delta_bias.unsqueeze(0).unsqueeze(-1))
    dA = torch.exp(torch.einsum("bdl,dn->bldn", dt, A))
    dB = torch.einsum("bdl,bnl->bldn", dt, B.squeeze(1))
    AttnMatrixOverCLS = torch.zeros((x_shape[0], x_shape[1], x_shape[2], x_shape[2]),requires_grad=True).to(dtype).to(dA.device) #BHLL: L vectors per batch and channel
    for r in range(L):
        for c in range(r+1):
            curr_C = C[:,:,:,r]
            currA = torch.ones((dA.shape[0],dA.shape[2],dA.shape[3]),requires_grad=True, dtype = dtype).to(dA.device)
            if c < r:
                for i in range(r-c):
                    currA = currA*dA[:,r-i,:,:]
            currB = dB[:,c,:,:]
            AttnMatrixOverCLS[:,:,r,c] = torch.sum(curr_C*currA*currB, axis=-1)
    return AttnMatrixOverCLS   


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X
def cross_selective_scan(
    x: torch.Tensor=None, 
    f: torch.Tensor=None,
    x_proj_weight: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    delta_softplus = True,
    out_norm: torch.nn.Module=None,
    # ==============================
    to_dtype=True, # True: final out to dtype
    # ==============================
    nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;


):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
    x = torch.cat((f, x), dim=-1)
    B, D, L = x.shape
    D, N = A_logs.shape
    D, R = dt_projs_weight.shape

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
        
    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1
            
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)
    
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=-1)
    dts = dt_projs_weight @ dts.t()
    dts = rearrange(dts, "d (b l) -> b d l", l=L)
    Bs = rearrange(Bs, "(b l) dstate -> b dstate l", l=L).contiguous()
    Cs = rearrange(Cs, "(b l) dstate -> b dstate l", l=L).contiguous()
    

    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)

    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.to(torch.float)

    ys = selective_scan_fn(
        x,
        dts,
        As,
        Bs,
        Cs,
        Ds,
        z=None,
        delta_bias=delta_bias,
        delta_softplus=True,
        return_last_state=None,
        )
    
    # return ys

    # if out_norm_shape in ["v1"]: # (B, C, H, W)
    #     y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
    # else: # (B, L, C)
    Bs = rearrange(Bs, "b dstate l -> b 1 dstate l")
    Cs = rearrange(Cs, "b dstate l -> b 1 dstate l")
    xai_matrix = compute_attn_matrix_fn(dts, delta_bias, A=As, B=Bs, C=Cs, L=L, x_shape=x.shape)
    # print('xai_matrix',xai_matrix.shape)
    y = rearrange(ys[:,:,1:], "b d l -> b l d")
    y = out_norm(y)

    return (y.to(x.dtype) if to_dtype else y), xai_matrix
class drqssm(nn.Module):
    def __init__(
        self, 
        d_model=96,
        d_state=16,
        d_conv=4,
        bias=False,
        device=None,
        dtype=None,
        ssm_ratio=2.0,
        dt_rank="auto",
        dropout=0.0,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        conv_bias=True,
        dt_init_floor=1e-4,
        ):
        super().__init__()
        print('newmamba')
        factory_kwargs = {"device": device, "dtype": dtype}
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        # in proj =======================================

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        # x proj ============================
        self.x_proj = nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)

        # dt proj ============================
        self.dt_projs = self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)

        
        # softmax | sigmoid | dwconv | norm ===========================

        self.out_norm = nn.LayerNorm(d_inner)
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, merge=True) # (K * D)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def forward_core(self, x: torch.Tensor, f,cross_selective_scan=cross_selective_scan):
        return cross_selective_scan(
            x, f, self.x_proj.weight, self.dt_projs.weight, self.dt_projs.bias.float(),
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=self.out_norm,
        )
    def forward(self, x: torch.Tensor, f,  **kwargs):

        # x = causal_conv1d_fn(
        #     x=x.permute(0, 2, 1),
        #     weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
        #     bias=self.conv1d.bias,
        #     activation="silu",
        # )
        y = self.forward_core(x, f)

        # if not self.disable_z:
        return y


class FSmamba(nn.Module):
    def __init__(
        self, 
        d_model=96,
        d_state=16,
        d_conv=4,
        bias=False,
        device=None,
        dtype=None,
        ssm_ratio=2.0,
        act_layer=nn.SiLU,
        dt_rank="auto",
        dropout=0.0,
        ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        # in proj =======================================
        d_proj = d_inner * 2
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # self.ffc =  nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)

        # nn.init.kaiming_normal_(self.ffc.weight, a=0, mode='fan_out')
        # ssm ======================================
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.fssm = drqssm(d_model=d_model,
                           d_state=d_state,
                           d_conv=d_conv,
                           ssm_ratio=ssm_ratio)
        
        self.prompt_proj = nn.Linear(512, d_inner,)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        self.fpool = nn.AdaptiveAvgPool2d((1, 1))
        # out proj =======================================
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
    def _cross_attention(self, query, context, mask=None, **kwargs):
        # print(query.shape)
        # print(context.shape)
        cross_weights = torch.matmul(query, context.permute(0, 2, 1))

        if mask is not None:
            cross_weights = cross_weights * mask.unsqueeze(1)

        cross_weights = l1norm(torch.relu(cross_weights), dim=-1)
        # cross_weights += torch.eye(cross_weights.size(-1)).to(cross_weights.device)
        
        wcontext = torch.matmul(cross_weights, context)

        return wcontext, cross_weights
    def forward(self, x, f):
        # f = f.detach()
        f = self.prompt_proj(self.fpool(f).flatten(1)).unsqueeze(dim=-1)
        # fe = f
        # # print(fe.shape)
        # f = self.fpool(f).flatten(1).unsqueeze(dim=-1)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = causal_conv1d_fn(
            x=x.permute(0, 2, 1),
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation="silu",
        )
        xff = x
        xbb = x.flip([-1])
        yf, af = self.fssm(xff, f)
        # yf = rearrange(yf[:,:,1:], "b d l -> b l d")
        yb, ab = self.fssm(xbb, f)
        # yb = rearrange(yb[:,:,1:], "b d l -> b l d")
        # res, _ = self._cross_attention(z, fe.flatten(2).transpose(1, 2))
        
        out  = (yf+yb.flip([1]))*z
        print(out.shape)
        print(f.transpose(-1, -2).shape)
        out = self.dropout(self.out_proj(out))+f
        return out
