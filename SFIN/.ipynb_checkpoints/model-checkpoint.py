import torch.nn as nn
import torch
from secondary_models_end import Backbone, LSET, CIAM


class SFIN(nn.Module):

    def __init__(self, net='resnet18', numclass=17, h=4):
        super().__init__()

        self.feature_extractor = Backbone(net)
        self.lset1 = LSET(dim=512, nbclass=numclass, head=h)
        self.lset2 = LSET(dim=256, nbclass=numclass, head=h)
        self.ciam = CIAM()
        self.classifier1 = nn.Conv2d(512, numclass, 1)
        self.classifier2 = nn.Conv2d(256, numclass, 1)
        self.fc = nn.Linear(256 + 512, 256 + 512)
        print('end')
    def forward(self, x):
        features = self.feature_extractor(x)

        F1 = features[-1]
        F2 = features[-2]

        g1, sf1 = self.lset1(F1)
        g2, sf2 = self.lset2(F2)
        F1_2, F2_2 = self.ciam(F1_1, F2_1)
        # F1_2, F2_2 = F1, F2

        out1 = self.classifier1(F1_2)
        final_class1 = torch.mean(out1, dim=(2, 3))
        out2 = self.classifier2(F2_2)
        final_class2 = torch.mean(out2, dim=(2, 3))
        final_class = (final_class1 + final_class2) / 2
        sf = torch.cat([sf1, sf2], dim=2)
        sf = self.fc(sf)

        #return g1, g2, sf, final_class, F1, F2, F1_1, F2_1
        return g1, g2, sf, final_class
