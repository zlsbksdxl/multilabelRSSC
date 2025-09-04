import torch.nn as nn
import torchvision
import torch
from einops import rearrange


class Backbone(nn.Module):

    def __init__(self, net):
        super().__init__()

        if net == 'resnet18':
            self.backbone = torchvision.models.resnet18(pretrained=True)
        else:
            raise ValueError('模型错误')

    def forward(self, x):
        f = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        f.append(x)
        x = self.backbone.layer4(x)
        f.append(x)
        return f


class SSAM(nn.Module):
    '''
    输入feature, 输出用于监督的sup_feature和语义信息semantic_feature
    '''

    def __init__(self, dim=512, nbclass=17) -> None:
        super().__init__()
        self.convclass = nn.Conv2d(dim, nbclass, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.nbclass = nbclass

    def forward(self, x):
        g = self.pool(x)
        g = self.convclass(g)
        g = g.view(g.size(0), -1)  #用于监督的sup_feature
        cam = self.convclass(x)
        x_ = x.repeat(self.nbclass, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        cam_ = torch.sigmoid(cam.unsqueeze(2))
        sf = torch.mean(cam_ * x_, dim=(3, 4))
        return g, sf


class Attention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)

    def forward(self, x, sf):
        Bx, Cx, Hx, Wx = x.shape
        q = x.flatten(2).transpose(2, 1)
        B, N, C = q.shape
        q = self.q(q).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        Bkv, Nkv, Ckv = sf.shape
        kv = self.kv(sf)
        kv = kv.reshape(Bkv, Nkv, 2, self.num_heads,
                        Ckv // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn @ v
        attn = rearrange(attn,
                         '(b) h (hh ww) d -> b (h d) (hh) (ww)',
                         h=self.num_heads,
                         d=Cx // self.num_heads,
                         hh=Hx,
                         ww=Wx)
        return attn


class FFN(nn.Module):

    def __init__(self, dim, act_layer=nn.ReLU6, drop=0.):
        super(FFN, self).__init__()
        out_features = dim
        hidden_features = int(dim * 4)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, 1),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU6(),
            nn.Conv2d(hidden_features, out_features, 1),
            nn.BatchNorm2d(out_features),
        )

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return x


class LSET(nn.Module):

    def __init__(self, dim=512, nbclass=17, head=4):
        super().__init__()
        self.ssam = SSAM(dim=dim, nbclass=nbclass)
        self.attn = Attention(dim=dim, num_heads=head)
        self.drop_path = nn.Identity()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.mlp = FFN(dim)

    def forward(self, x):
        sup_g, sf = self.ssam(x)
        x_ = self.attn(x, sf)
        x = x + self.drop_path(self.norm1(x_))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return sup_g, sf, x


class CIAM(nn.Module):
    '''
    输入两个特征图，完成空间注意力后，输出两个特征图
    '''

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 1, 1)
        self.conv2 = nn.Conv2d(2, 1, 1)
        self.down = nn.AdaptiveAvgPool2d((8,8))
        self.up = nn.AdaptiveAvgPool2d((16,16))

    def forward(self, x1, x2):

        max1, _ = torch.max(x1, dim=1, keepdim=True)
        avg1 = torch.mean(x1, dim=1, keepdim=True)
        attn1 = self.conv1(torch.cat([max1, avg1], dim=1))

        max2, _ = torch.max(x2, dim=1, keepdim=True)
        avg2 = torch.mean(x2, dim=1, keepdim=True)
        attn2 = self.conv1(torch.cat([max2, avg2], dim=1))

#         attn1_ = self.sigmoid(attn1 + self.down(attn2))
        attn1_ = self.sigmoid(attn1) + self.sigmoid(self.down(attn2))
#         attn2_ = self.sigmoid(attn2 + self.up(attn1))
        attn2_ = self.sigmoid(self.up(attn1)) + self.sigmoid(attn2)

        return attn1_ * x1, attn2_ * x2

