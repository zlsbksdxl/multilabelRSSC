import torch
import torch.nn as nn
import sys
import numpy as np
from model_auxiliary import FBackbone, GroupWiseLinear
from feature_branch import Pyramidmamba
from semantic_branch import FSmamba




class mlmodle(nn.Module):
    def __init__(self,  backbone_name='resnet18', num_class=17, hidden_dim=128):
        super().__init__()
        print('该模型,先利用金字塔mamba增强全局信息,然后使用特征引导的语义增强mamba增强语义完成分类')
        #----------------backbone------------------
        self.backbone = FBackbone(net=backbone_name)
        
        # ---------------全局信息增强------------------
        self.feature_enhance = Pyramidmamba()
        # self.feature_enhance1 = Pyramidmamba()
        # self.feature_enhance2 = Pyramidmamba()
        #---------------语义分支--------------------
        self.semantic_enhance = FSmamba(hidden_dim)
        # self.semantic_enhance1 = FSmamba(hidden_dim)
        # self.semantic_enhance2 = FSmamba(hidden_dim)
        self.query_embed = nn.Parameter(torch.zeros(
                1, num_class, hidden_dim))
        #self.query_embed = torch.from_numpy(np.load('/data/newdataset/ucm.npy')).float().cuda()
       
        # val = math.sqrt(6. / float(3 * reduce(mul, _pair(17), 1) + 128))  # noqa
        # nn.init.uniform_(self.query_embed.data, -val, val)
        #---------------完成分类--------------------
        self.classifier = GroupWiseLinear(num_class, hidden_dim, bias=True)

    def forward(self, image):
        b, c, h, w = image.shape
        features = self.backbone(image)
        
        features_ = self.feature_enhance(features)
        # features_ = self.feature_enhance1(features_)
        # features_ = self.feature_enhance2(features_)
        query_input = self.query_embed
        query_input = query_input.expand(b, -1, -1)
        query_output= self.semantic_enhance(query_input, features_)
        # query_output = self.semantic_enhance1(query_output, features_)
        # query_output = self.semantic_enhance2(query_output, features_)
        
        
        # cla_features = self.feature_enhance(features, query_input)
        
        output = self.classifier(query_output)
        # final_class = torch.mean(output, dim=(2, 3))
        return output
    
    
