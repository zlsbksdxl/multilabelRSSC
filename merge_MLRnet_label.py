import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置路径
LABELS_DIR = "/root/autodl-fs/MLRSnet/labels"  # 修改为你的labels目录
OUTPUT_CSV = "/root/autodl-fs/MLRSnet/mlrsnet_multilabels.csv"
OUTPUT_NPY = "/root/autodl-tmp/multilabelRSSC/SFIN/mlrsnet_ml.npy"

# 识别图片名列的候选（大小写不敏感）
IMAGE_COL_CANDIDATES = {
    "image", "image_name", "imagename", "imageid", "image_id",
    "name", "filename", "file", "mage_name", "mage"  # 包含你出现的 MAGE_NAME 拼写
}

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # 去除列名两端空白，并统一小写对比时不改变原名，仅内部查找使用
    df.columns = [c.strip() for c in df.columns]
    return df

def detect_image_col(columns):
    # 返回原列名
    for c in columns:
        cl = c.strip().lower()
        if cl in IMAGE_COL_CANDIDATES or "image" in cl or "name" in cl:
            return c
    raise ValueError(f"未找到图片名列，请检查CSV表头: {columns}")

def to_zero_one(x):
    # 将单元格转为0/1
    if pd.isna(x):
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    try:
        v = int(float(s))
        return 1 if v != 0 else 0
    except:
        # 若出现 "True"/"False" 等
        return 1 if s.lower() in {"1", "true", "yes"} else 0

def build_category_order(label_files):
    categories = []
    seen = set()
    for lf in label_files:
        df = pd.read_csv(os.path.join(LABELS_DIR, lf), encoding="utf-8-sig")
        df = normalize_cols(df)
        img_col = detect_image_col(df.columns)
        file_cats = [c for c in df.columns if c != img_col]
        # 按文件中的顺序追加新的类别
        for c in file_cats:
            if c not in seen:
                seen.add(c)
                categories.append(c)
    return categories

def merge_labels():
    files = [f for f in os.listdir(LABELS_DIR) if f.lower().endswith(".csv")]
    files.sort()
    if not files:
        raise RuntimeError(f"{LABELS_DIR} 下未找到CSV文件")

    print(f"找到 {len(files)} 个标签文件")

    # 先统一类别顺序（支持不同文件头部存在细微差异的情况，取并集，按首次出现顺序）
    categories = build_category_order(files)
    print(f"类别数: {len(categories)}")
    # 用于聚合的map：key为图片名（含扩展名），value为np数组
    nm2vec = {}

    # 第二遍：逐文件合并
    for lf in tqdm(files, desc="合并标签"):
        path = os.path.join(LABELS_DIR, lf)
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = normalize_cols(df)
        img_col = detect_image_col(df.columns)

        # 当前文件中的列 -> 位置索引 映射
        cols_in_file = {c: idx for idx, c in enumerate(df.columns) if c != img_col}

        for _, row in df.iterrows():
            img_name_raw = str(row[img_col]).strip()
            if not img_name_raw or img_name_raw.lower() == "image":
                continue  # 跳过错误/标题行

            # 规范化图片名：确保包含扩展名（若无扩展名，默认补 .jpg）
            root, ext = os.path.splitext(img_name_raw)
            if ext == "":
                img_name = root + ".jpg"
            else:
                img_name = img_name_raw

            # 构建该行的向量（按全局categories顺序）
            vec = np.zeros(len(categories), dtype=np.int8)
            for j, cat in enumerate(categories):
                if cat in df.columns:
                    vec[j] = to_zero_one(row[cat])
                else:
                    vec[j] = 0

            # 聚合：同名图片做逐位 OR
            if img_name in nm2vec:
                nm2vec[img_name] = np.maximum(nm2vec[img_name], vec)
            else:
                nm2vec[img_name] = vec

    # 写CSV（表头为 image + categories）
    print(f"正在写入CSV: {OUTPUT_CSV}")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image"] + categories)
        for img_name, vec in nm2vec.items():
            w.writerow([img_name] + vec.astype(int).tolist())

    # 写NPY：键使用不带扩展名，值为float32向量；同时保存categories方便对齐
    print(f"正在写入NPY: {OUTPUT_NPY}")
    os.makedirs(os.path.dirname(OUTPUT_NPY), exist_ok=True)
    nm2label = {}
    for img_name, vec in nm2vec.items():
        nm2label[os.path.splitext(img_name)[0]] = vec.astype(np.float32)
    np.save(OUTPUT_NPY, {"nm2label": nm2label, "categories": categories})

    print(f"完成！图片数: {len(nm2vec)}, 维度: {len(categories)}")
    # 显示一个样例
    some_key = next(iter(nm2vec.keys()))
    print(f"示例: {some_key} -> {nm2vec[some_key].sum()} 个正标签")

if __name__ == "__main__":
    merge_labels()