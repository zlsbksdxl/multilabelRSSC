import glob
from collections import defaultdict
import os
import numpy as np
import random
from tqdm import tqdm
# from collections import Counter

# import torchvision.transforms as transforms

from PIL import Image

dir_pth = os.path.dirname(os.path.abspath(__file__))
UCM_ML_npy = os.path.join(dir_pth, 'ucm_ml.npy')
AID_ML_npy = os.path.join(dir_pth, 'mlrsnet_ml.npy')
UFC_ML_npy = os.path.join(dir_pth, 'dfc_ml.npy')  # 确保创建此文件

def default_loader(path):
    return Image.open(path).convert('RGB')

class FlatDataGenerator:
    """
    用于处理扁平目录结构数据集的生成器
    训练和测试图像在同一目录下，通过文件名前缀区分
    """

    def __init__(self,
                 datadir,
                 dataset,
                 imgExt='png',
                 imgTransform=None,
                 phase='train',
                 train_prefix='tr_',
                 test_prefix='te_'):
        self.dataset = dataset
        self.datadir = datadir
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.train_prefix = train_prefix
        self.test_prefix = test_prefix
        
        self.train_idx2fileDict = {}
        self.test_idx2fileDict = {}
        
        # 加载数据
        self.load_data()

    def load_data(self):
        """加载数据并区分训练/测试集"""
        # 加载标签数据
        if self.dataset == 'UCM':
            data = np.load(UCM_ML_npy, allow_pickle=True).item()
        elif self.dataset == 'AID':
            data = np.load(AID_ML_npy, allow_pickle=True).item()
        elif self.dataset == 'UFC':
            data = np.load(UFC_ML_npy, allow_pickle=True).item()
        else:
            raise ValueError(f"不支持的数据集: {self.dataset}")
            
        # 获取所有图像文件
        all_images = sorted(glob.glob(os.path.join(self.datadir, f'*.{self.imgExt}')))
        
        # 区分训练和测试图像
        train_images = [img for img in all_images if os.path.basename(img).startswith(self.train_prefix)]
        test_images = [img for img in all_images if os.path.basename(img).startswith(self.test_prefix)]
        
        self.train_numImgs = len(train_images)
        self.test_numImgs = len(test_images)
        
        # 填充训练集字典
        for i, imgPth in enumerate(train_images):
            # 从文件名中去除前缀以匹配标签数据
            base_name = os.path.basename(imgPth)
            img_name = base_name[len(self.train_prefix):].split('.')[0]
            
            if img_name in data['nm2label']:
                multi_hot = data['nm2label'][img_name]
                self.train_idx2fileDict[i] = (imgPth, multi_hot)
            else:
                print(f"警告: 找不到训练图像 {img_name} 的标签")
        
        # 填充测试集字典
        for i, imgPth in enumerate(test_images):
            base_name = os.path.basename(imgPth)
            img_name = base_name[len(self.test_prefix):].split('.')[0]
            
            if img_name in data['nm2label']:
                multi_hot = data['nm2label'][img_name]
                self.test_idx2fileDict[i] = (imgPth, multi_hot)
            else:
                print(f"警告: 找不到测试图像 {img_name} 的标签")
        
        print(f"数据集: {self.dataset}")
        print(f"训练图像数: {len(self.train_idx2fileDict)}")
        print(f"测试图像数: {len(self.test_idx2fileDict)}")
        
        # 创建索引列表
        self.trainDataIndex = list(range(len(self.train_idx2fileDict)))
        self.testDataIndex = list(range(len(self.test_idx2fileDict)))

    def __getitem__(self, index):
        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        elif self.phase == 'test':
            idx = self.testDataIndex[index]
        return self.__datagen(idx)

    def __datagen(self, idx):
        if self.phase == 'train':
            imgPth, multi_hot = self.train_idx2fileDict[idx]
        elif self.phase == 'test':
            imgPth, multi_hot = self.test_idx2fileDict[idx]
        img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)

        return {
            'img': img,
            'idx': idx,
            'multiHot': multi_hot.astype(np.float32),
        }
        
    def __len__(self):
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'test':
            return len(self.testDataIndex)

    
class DataGeneratorML:

    def __init__(self,
                 datadir,
                 dataset,
                 imgExt='jpg',
                 imgTransform=None,
                 phase='train',
                 train_ratio=0.8,
                 seed=123):
        self.dataset = dataset
        self.datadir = datadir
        self.sceneList = [
            os.path.join(self.datadir, x)
            for x in sorted(os.listdir(self.datadir))
            if os.path.isdir(os.path.join(self.datadir, x))
        ]  #每一类的路径
        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.seed = seed
        self.train_ratio = train_ratio
        self.CreateIdx2fileDict()

    def CreateIdx2fileDict(self):
        random.seed(self.seed)

        if self.dataset == 'UCM':
            data = np.load(UCM_ML_npy, allow_pickle=True).item()
        elif self.dataset == 'AID':
            data = np.load(AID_ML_npy, allow_pickle=True).item()

        self.train_numImgs = 0
        self.test_numImgs = 0
        train_count = 0
        test_count = 0
        for _, scenePth in enumerate(self.sceneList):
            subdirImgPth = sorted(
                glob.glob(os.path.join(scenePth, '*.' + self.imgExt)))
#             random.shuffle(subdirImgPth)
            train_subdirImgPth = subdirImgPth[:int(self.train_ratio * len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(self.train_ratio * len(subdirImgPth)):]
            for i in range(len(train_subdirImgPth)):
                for j in range(len(test_subdirImgPth)):
                    if train_subdirImgPth[i]==test_subdirImgPth[j]:
                        print('yes')
#             print(train_subdirImgPth[train_subdirImgPth==test_subdirImgPth])
#             print(train_subdirImgPth)
#             print(test_subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            
            for imgPth in train_subdirImgPth:
                multi_hot = data['nm2label'][os.path.basename(imgPth).split(
                    '.')[0]]
                self.train_idx2fileDict[train_count] = (imgPth, multi_hot)
                train_count += 1
            for imgPth in test_subdirImgPth:
                multi_hot = data['nm2label'][os.path.basename(imgPth).split(
                    '.')[0]]
                self.test_idx2fileDict[test_count] = (imgPth, multi_hot)
                test_count += 1
        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))

    def __getitem__(self, index):
        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        elif self.phase == 'test':
            idx = self.testDataIndex[index]
        return self.__datagen(idx)

    def __datagen(self, idx):
        if self.phase == 'train':
            imgPth, multi_hot = self.train_idx2fileDict[idx]
        elif self.phase == 'test':
            imgPth, multi_hot = self.test_idx2fileDict[idx]
        img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)

        return {
            'img': img,
            'idx': idx,
            'multiHot': multi_hot.astype(np.float32),
        }
    def __len__(self):

        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'test':
            return len(self.testDataIndex)

