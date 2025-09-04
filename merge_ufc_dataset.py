import os
import shutil
from tqdm import tqdm

# 定义源路径和目标路径
TRAIN_PATH = "/root/autodl-fs/UFC15-ML/images_tr"
TEST_PATH = "/root/autodl-fs/UFC15-ML/images_test"
TARGET_PATH = "/root/autodl-fs/UFC15-ML/merged"  # 合并后的目标路径

# 确保目标目录存在
os.makedirs(TARGET_PATH, exist_ok=True)

# 处理训练集图像
print("正在复制训练集图像...")
train_files = [f for f in os.listdir(TRAIN_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
for img in tqdm(train_files):
    src = os.path.join(TRAIN_PATH, img)
    # 添加tr_前缀以区分训练集图像
    dst = os.path.join(TARGET_PATH, f"tr_{img}")
    shutil.copy2(src, dst)

# 处理测试集图像
print("正在复制测试集图像...")
test_files = [f for f in os.listdir(TEST_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
for img in tqdm(test_files):
    src = os.path.join(TEST_PATH, img)
    # 添加te_前缀以区分测试集图像
    dst = os.path.join(TARGET_PATH, f"te_{img}")
    shutil.copy2(src, dst)

print(f"\n合并完成！所有图像已复制到 {TARGET_PATH}")
print(f"已复制 {len(train_files)} 张训练图像和 {len(test_files)} 张测试图像")
print("现在您可以将这个合并后的目录作为 datadir 参数传递给 DataGeneratorML 类")