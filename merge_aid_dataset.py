import os
import shutil
from tqdm import tqdm

# 定义源路径和目标路径
TRAIN_PATH = "/root/autodl-fs/UFC15-ML/images_tr"
TEST_PATH = "/root/autodl-fs/UFC15-ML/images_test"
TARGET_PATH = "/root/autodl-fs/UFC15-ML/merged"  # 合并后的目标路径

# 确保目标目录存在
os.makedirs(TARGET_PATH, exist_ok=True)

# 获取所有类别（假设训练集和测试集有相同的类别）
categories = [d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))]

print(f"找到 {len(categories)} 个类别，开始合并...")

# 对每个类别进行处理
for category in tqdm(categories, desc="处理类别"):
    # 创建目标类别目录
    category_target_dir = os.path.join(TARGET_PATH, category)
    os.makedirs(category_target_dir, exist_ok=True)
    
    # 处理训练集中的图像
    train_category_dir = os.path.join(TRAIN_PATH, category)
    for img in os.listdir(train_category_dir):
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            src = os.path.join(train_category_dir, img)
            # 添加前缀避免潜在的文件名冲突
            dst = os.path.join(category_target_dir, f"{img}")
            shutil.copy2(src, dst)
    
    # 处理测试集中的图像
    test_category_dir = os.path.join(TEST_PATH, category)
    for img in os.listdir(test_category_dir):
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            src = os.path.join(test_category_dir, img)
            # 添加前缀避免潜在的文件名冲突
            dst = os.path.join(category_target_dir, f"{img}")
            shutil.copy2(src, dst)

print(f"\n合并完成！所有图像已复制到 {TARGET_PATH}")
print("现在您可以将这个合并后的目录作为 datadir 参数传递给 DataGeneratorML 类")