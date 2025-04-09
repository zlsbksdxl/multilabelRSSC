import torch.nn as nn
import torchvision
import torch
from timm.models.layers import DropPath
import torch.nn.functional as F
# from newmodel1.model_semantic_auxiliary import drqssm
# from model_semantic_auxiliary import drqssm
from einops import rearrange, repeat
'''
使用金字塔mamba捕捉全局信息
'''

import math
from causal_conv1d import causal_conv1d_fn
import torch.nn.functional as F

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
# pytorch cross scan =============
def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X
def cross_selective_scan(
    x: torch.Tensor=None, 
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
    y = rearrange(ys, "b d l -> b l d")
    y = out_norm(y)

    return (y.to(x.dtype) if to_dtype else y)
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
        # print('newmamba')
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
    
    def forward_core(self, x: torch.Tensor,cross_selective_scan=cross_selective_scan):
        return cross_selective_scan(
            x, self.x_proj.weight, self.dt_projs.weight, self.dt_projs.bias.float(),
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=self.out_norm,
        )
    def forward(self, x: torch.Tensor,  **kwargs):

        y = self.forward_core(x)

        return y
def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

class Pyramidmamba(nn.Module):
    def __init__(self, inputdim=512, hiddendim=128):
        super().__init__()
        d_inner = hiddendim*2
        d_proj = d_inner * 2
        #-----多尺度卷积+降维-----
        self.in_proj = nn.Conv2d(inputdim, d_proj, 1, 1)
        self.out_proj = nn.Conv2d(d_inner, inputdim, 1, 1)
        self.conv1 = nn.Conv2d(d_inner, d_inner, 1, 1)
        self.conv2 = nn.Conv2d(d_inner, d_inner, 2, 2)
        self.conv3 = nn.Conv2d(d_inner, d_inner, 2, 2)
        #-----全局信息建模-----
        self.fssm1 = drqssm(d_model=hiddendim,
                           d_state=16,
                           d_conv=4,
                           ssm_ratio=2)
        self.fssm2 = drqssm(d_model=hiddendim,
                           d_state=16,
                           d_conv=4,
                           ssm_ratio=2)
        self.fssm3 = drqssm(d_model=hiddendim,
                           d_state=16,
                           d_conv=4,
                           ssm_ratio=2)
        
        self.act = nn.SiLU()
    def _cross_attention(self, query, context, mask=None, **kwargs):
        query = query.flatten(2).transpose(1, 2)
        cross_weights = query @ context.permute(0, 2, 1)

        if mask is not None:
            cross_weights = cross_weights * mask.unsqueeze(1)

        cross_weights = cross_weights.softmax(dim=-1)
        # cross_weights += torch.eye(cross_weights.size(-1)).to(cross_weights.device)
        # print(cross_weights.shape)
        wcontext = cross_weights @ context
        return wcontext
    def forward(self, input_f):
        xz = self.in_proj(input_f)
        x, z = xz.chunk(2, dim=1)
        #----自下而上----
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        #----自上而下----
        b3, c3, h3, w3 = f3.shape
        # print(f3.flatten(2).shape)
        yf3 = self.fssm3(f3.flatten(2))
        # print(yf3.shape)
        yb3 = self.fssm3(f3.flatten(2).flip([-1]))
        y3 = yf3 + yb3.flip([1])
        y2_ = F.interpolate(rearrange(y3, "b (h w) c -> b c h w", h=h3, w=w3), scale_factor=2).flatten(2).transpose(1, 2)
        
        b2, c2, h2, w2 = f2.shape
        yf2 = self.fssm2(f2.flatten(2))
        yb2 = self.fssm2(f2.flatten(2).flip([-1]))
        y2 = yf2 + yb2.flip([1]) + y2_
        y1_ = F.interpolate(rearrange(y2, "b (h w) c -> b c h w", h=h2, w=w2), scale_factor=2).flatten(2).transpose(1, 2)
        
        b1, c1, h1, w1 = f1.shape
        yf1 = self.fssm1(f1.flatten(2))
        yb1 = self.fssm2(f1.flatten(2).flip([-1]))
        y1 = yf1 + yb1.flip([1])+ y1_
        out1 = rearrange(y1, "b (h w) c -> b c h w", h=h1, w=w1)
        out = (out1)*self.act(z)
        out = self.out_proj(out) #+ input_f
        return out
if __name__=='__main__':

    x = torch.rand(2, 512, 8, 8).cuda()
    # sf  = torch.rand(2, 17, 128).cuda()
    model = Pyramidmamba().cuda()
    y = model(x)
    print(y.shape)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('参数量', n_parameters/1000000)
