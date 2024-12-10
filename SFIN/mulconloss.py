
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class mulConLoss(nn.Module):


    def __init__(self, device, temperature=0.5):
        super(mulConLoss, self).__init__()
        self.temperature = temperature
        self.dev = device

    def forward(self, features, labels=None, mask=None):


        # 选择参与训练的特征
        '''
        features [b c dim]
        labels [b c]
        '''
        nonzero_indices = torch.nonzero(labels == 1, as_tuple=True)
        selected_features = features[nonzero_indices[0],
                                     nonzero_indices[1], :]  #n*dim
        selected_labels = nonzero_indices[1]  #n*1
        # 生成mask
        selected_labels = selected_labels.contiguous().view(-1, 1)
        if selected_labels.shape[0] != selected_features.shape[0]:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(selected_labels,
                        selected_labels.T).float().to(self.dev)
        n = selected_labels.shape[0]

        #计算

        similarity_matrix = F.cosine_similarity(selected_features.unsqueeze(1),
                                                selected_features.unsqueeze(0),
                                                dim=2)  #相似度矩阵

        mask_sim = mask  #正对的mask
        mask_no_sim = torch.ones_like(mask) - mask_sim  #负对的mask
        mask_dui_jiao_0 = torch.ones_like(mask) - torch.eye(n, n).to(self.dev)  #对角线为0
        similarity_matrix = torch.exp(similarity_matrix /
                                      self.temperature)  #相似度矩阵
        similarity_matrix = similarity_matrix * mask_dui_jiao_0  #不与自己计算
        sim = mask_sim * similarity_matrix  #正对的相似度矩阵
        non_sim = similarity_matrix - sim
        no_sim_sum = torch.sum(non_sim, dim=1)
        no_sim_expend = no_sim_sum.repeat(n, 1).T  #负对的相似对和
        sim_sum = sim + no_sim_expend  #分母
        loss = torch.div(sim, sim_sum)
        loss = mask_no_sim + loss + torch.eye(n, n).to(self.dev)  #防止log0
        loss = -torch.log(loss)

        #  计算均值1
#         non_zero_mask = (loss != 0)  # 使用布尔掩码找到非零元素
#         loss = torch.sum(loss * non_zero_mask, dim=1)
#         non_zero_count_per_row = torch.sum(non_zero_mask, dim=1)
#         loss = loss / non_zero_count_per_row
#         loss[non_zero_count_per_row == 0] = 0.0
#         loss = torch.sum(loss)

        #  计算均值2
        loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

        return loss
