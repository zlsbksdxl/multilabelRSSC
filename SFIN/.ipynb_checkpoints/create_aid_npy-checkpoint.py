import numpy as np
import os
import csv
from tqdm import tqdm

# --- 配置 ---
# CSV文件路径
INPUT_CSV_PATH = r'/root/autodl-fs/UFC15-ML/multilabel.csv'

# 输出文件名
OUTPUT_NPY_FILE = 'aid1_ml.npy'

# --- 主逻辑 ---
def create_npy_from_csv(csv_path, output_path):
    """
    解析CSV文件并创建aid_ml.npy
    """
    print(f"正在读取CSV标签文件: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        print("请检查 INPUT_CSV_PATH 变量是否设置正确。")
        return

    nm2label_dict = {}
    
    with open(csv_path, 'r', newline='') as f:
        # 使用csv模块读取CSV文件
        csv_reader = csv.reader(f)
        
        # 读取标题行
        header = next(csv_reader, None)
        if header:
            print(f"文件标题行: {','.join(header)}")
        
        # 处理数据行
        print("正在处理图像标签...")
        for row in tqdm(csv_reader):
            if not row or len(row) < 2:
                continue
            
            # 第一列是图像名称
            image_name = row[0]
            
            # 其余列是标签 (0 或 1)
            try:
                labels = np.array([int(label) for label in row[1:]], dtype=np.float32)
                nm2label_dict[image_name] = labels
            except ValueError:
                print(f"警告: 跳过行 {row} - 包含非整数标签值")

    # 创建 datagen.py 需要的最终字典结构
    data_to_save = {'nm2label': nm2label_dict}

    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_full_path = os.path.join(current_dir, output_path)

    print(f"\n正在保存 .npy 文件到: {output_full_path}")
    np.save(output_full_path, data_to_save)

    print("\n处理完成!")
    print(f"总共处理了 {len(nm2label_dict)} 个图像的标签。")
    print(f"文件 '{output_path}' 已成功生成在当前目录下。")
    print("现在您可以继续运行主训练脚本了。")


if __name__ == '__main__':
    create_npy_from_csv(INPUT_CSV_PATH, OUTPUT_NPY_FILE)