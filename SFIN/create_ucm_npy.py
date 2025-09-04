import numpy as np
import os
from tqdm import tqdm

# --- 配置 ---
# 请将此路径修改为您 "LandUse_Multilabeled.txt" 文件的实际路径
# 根据您之前提供的信息，它应该在这里：
INPUT_TXT_PATH = r'/root/autodl-fs/UFC15-ML/multilabel.csv'

# 输出文件名，datagen.py 将加载此文件
OUTPUT_NPY_FILE = 'aid1_ml.npy'

# --- 主逻辑 ---
def create_npy_from_txt(txt_path, output_path):
    """
    解析 LandUse_Multilabeled.txt 文件并创建 ucm_ml.npy
    """
    print(f"正在读取标签文件: {txt_path}")
    if not os.path.exists(txt_path):
        print(f"错误: 找不到文件 {txt_path}")
        print("请检查 INPUT_TXT_PATH 变量是否设置正确。")
        return

    nm2label_dict = {}
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
        # 跳过第一行标题行
        header = lines[0]
        print(f"文件标题行: {header.strip()}")
        
        # 从第二行开始处理
        print("正在处理图像标签...")
        for line in tqdm(lines[1:]):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            # 第一个元素是图像名 (例如 'agricultural00')
            image_name = parts[0]
            
            # 后面的元素是标签 (0 或 1)
            labels = np.array([int(l) for l in parts[1:]], dtype=np.float32)
            
            nm2label_dict[image_name] = labels

    # 创建 datagen.py 需要的最终字典结构
    data_to_save = {'nm2label': nm2label_dict}

    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_full_path = os.path.join(current_dir, output_path)

    print(f"\n正在保存 .npy 文件到: {output_full_path}")
    np.save(output_full_path, data_to_save)

    print("\n处理完成!")
    print(f"总共处理了 {len(nm2label_dict)} 个图像的标签。")
    print(f"文件 '{output_path}' 已成功生成在 SFIN 目录下。")
    print("现在您可以继续运行主训练脚本了。")


if __name__ == '__main__':
    create_npy_from_txt(INPUT_TXT_PATH, OUTPUT_NPY_FILE)