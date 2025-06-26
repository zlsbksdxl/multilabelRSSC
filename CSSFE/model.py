import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse
import clip
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
from timm.models.layers import DropPath

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip.clip

out2_neck = False

import timm
import torch
import torch.nn as nn

import timm
import torch
import torch.nn as nn

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', img_size=448, pretrained=True, out_channels=2048):

        super(ViTFeatureExtractor, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, img_size=img_size)
        self.model.reset_classifier(0)
        
        if hasattr(self.model, 'patch_embed'):
            self.patch_size = self.model.patch_embed.patch_size[0]
            hidden_dim = self.model.patch_embed.proj.out_channels  
            self.patch_size = 16
            hidden_dim = 768  


        self.grid_size = img_size // self.patch_size

   
        self.conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        

        self.model.eval()

    def forward(self, x):
 
        tokens = self.model.forward_features(x)
       
        patch_tokens = tokens[:, 1:, :]  # 形状 (B, num_patches, hidden_dim)
       
        feature_map = patch_tokens.reshape(x.size(0), self.grid_size, self.grid_size, -1)
     
        feature_map = feature_map.permute(0, 3, 1, 2)
       
        feature_map = self.conv(feature_map)
        return feature_map


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if out2_neck:
            self.to_out = nn.Sequential(
                # nn.Linear(inner_dim, dim),
                nn.Linear(inner_dim, 2*inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2*inner_dim, dim),
                nn.Dropout(dropout)
                ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

    # @get_local('attn')
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

class SemanticFeatureCorrector(nn.Module):

    def __init__(self, d=2048, h=14, w=14):
        super().__init__()
        self.d = d
        self.h = h
        self.w = w
        self.mask = torch.load('/media/drq/coco/mask.pt')
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        
        self.adjust_generator = nn.Sequential(
            nn.Linear(d, h * w),
            nn.Sigmoid()
        )

    def forward(self, features, sem_vectors):
        # features: (b, c, h, w)
        # sem_vectors: (b, c, d)
        b, c, h, w = features.shape
        
        Q = self.W_q(sem_vectors)  # (b, c, d)
        K = self.W_k(sem_vectors)  # (b, c, d)
        
        S = torch.einsum('bcd,bkd->bck', Q, K) / (self.d ** 0.5)  # (b, c, c)
        S = F.softmax(S*self.mask, dim=2)
        
        features_flat = features.view(b, c, -1)  # (b, c, h*w)
        
        context_flat = torch.einsum('bck,bkh->bch', S, features_flat)  # (b, c, h*w)
        context = context_flat.view(b, c, h, w)  # (b, c, h, w)
        
        fused = features + context  # (b, c, h, w)
        
        adjust = self.adjust_generator(sem_vectors)  # (b, c, h*w)
        adjust = adjust.view(b, c, h, w)  # (b, c, h, w)
        
        corrected = fused * adjust  # (b, c, h, w)
        
        return corrected


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)

class model77(torch.nn.Module):

    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device = torch.device('cuda' if torch.cuda.is_available() else "cpu"))
        return prompts
    
    def __init__(self, cfg, classnames):
        super(model77, self).__init__()
        self.cfg = cfg
        clip_model, preprocess = clip.clip.load('/media/drq/e124a4b0-ba3f-4609-98e6-c91806920498/pretrain_models/clip/RN50.pt', device='cpu', jit=False)
        if cfg.backbone == 'resnet101':
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            # model = torchvision.models.vgg16(pretrained=True)
            
            # model.load_state_dict(torch.load(cfg.pretrained))
            feature_extractor = self.Backbone = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
            self.feat_dim=2048
            feat_dim=2048
        elif cfg.backbone == 'vit16':
            feature_extractor = ViTFeatureExtractor()
            self.feat_dim=2048
            feat_dim=2048
        self.attention = Transformer(
            dim=feat_dim,
            depth=1,
            heads=8,
            dim_head=512,
            mlp_dim=2048,
            dropout=0.1
        )
        self.prompts = self.get_tokenized_prompts(classnames)
        self.text_encoder = TextEncoder(clip_model)
        self.label_emb = nn.Sequential(
                            nn.Linear(1024, feat_dim*2),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(feat_dim*2, feat_dim),
                            nn.Dropout(0.1)
                        )

        self.feature_extractor = feature_extractor
        self.avgpool = GlobalAvgPool2d()

        self.onebyone_conv = nn.Conv2d(feat_dim, cfg.num_classes, 1)
        self.conv1d = nn.Conv1d(feat_dim, cfg.num_classes, 1)
        self.conv1d1 = nn.Conv1d(feat_dim, cfg.num_classes, 1)
        self.SFC = SemanticFeatureCorrector()
    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    def weighted_feature_aggregation(self, features, cams):
        # features: (b, c, h, w)
        # cams: (b, n, h, w)
        weighted_features = features.unsqueeze(1) * cams.unsqueeze(2)  # (b, n, c, h, w)
        aggregated = torch.mean(weighted_features, dim=(3,4))  # (b, n, c)
        return aggregated  # (b, n, c)
    def forward(self, x):
        batch_size = x.size(0)
        prompts = self.prompts
        text_features = self.text_encoder(prompts)
        text_features = text_features.float()
        attr = self.label_emb(text_features).unsqueeze(0).expand(batch_size, text_features.size(0), self.feat_dim)

        
        feats = self.feature_extractor(x)
        
        CAM = self.onebyone_conv(feats)
        CAM = self.SFC(CAM, attr)
        # print(CAM.shape)
        # CAM = torch.where(CAM > 0, CAM * self.alpha, CAM) # BoostLU operation

        f = self.weighted_feature_aggregation(feats, CAM)
        feature = torch.cat((f, attr), 1)
        feature = self.attention(feature)

        classify_feature = feature[:, -attr.size(1):, :]

        classify_feature = classify_feature.unsqueeze(-2)
        attr = attr[0].unsqueeze(-1)
        logits = torch.matmul(classify_feature, attr).squeeze()
        # logits = self.conv1d(classify_feature.permute(0, 2, 1)).permute(0, 2, 1)
        return logits
    
    
    
