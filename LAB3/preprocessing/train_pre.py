# -*- coding: utf-8 -*-
"""
分類影像 + 影像前處理 (Train / Val 自動處理版)
---------------------------------------------------
會自動：
1. 讀取 train.csv 與 val.csv
2. 根據 one-hot 標籤建立類別資料夾
3. 進行影像前處理 (Resize + CLAHE + Normalize)
4. 儲存到 sorted_preprocessed_train / sorted_preprocessed_val
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm

# ======== 路徑設定 ========
CSV_LIST = ["train_data.csv"]   # 同層放 train.csv / val.csv
IMG_DIR  = "./train_images"                   # 原始圖片資料夾
IMG_SIZE = (512, 512)                 # Resize 尺寸
APPLY_CLAHE = True                    # 是否啟用 CLAHE
NORMALIZE = True                      # 是否標準化到 [0,1]
CLASS_COLS = ["normal", "bacteria", "virus", "COVID-19"]

# ======== CLAHE 函數 ========
def apply_clahe_color(img):
    """對彩色影像進行 CLAHE 增強"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ======== 主流程函數 ========
def process_csv(csv_path, out_dir):
    print(f"\n📂 開始處理 {csv_path} → {out_dir}")

    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    for c in CLASS_COLS:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row["new_filename"]
        src_path = os.path.join(IMG_DIR, fname)

        # 找出該影像的類別
        class_name = None
        for c in CLASS_COLS:
            if row[c] == 1:
                class_name = c
                break

        if class_name is None:
            print(f"[Warning] No class found for {fname}")
            continue
        if not os.path.exists(src_path):
            print(f"[Missing] {src_path}")
            continue

        # === 讀取與前處理 ===
        img = cv2.imread(src_path)
        if img is None:
            print(f"[Error] Cannot read {src_path}")
            continue

        # 儲存
        dst_dir = os.path.join(out_dir, class_name)
        dst_path = os.path.join(dst_dir, fname)
        cv2.imwrite(dst_path, img)

    print(f"✅ {csv_path} 已完成前處理與分類！")

# ======== 主程式入口 ========
if __name__ == "__main__":
    for csv_name in CSV_LIST:
        if not os.path.exists(csv_name):
            print(f"⚠️ 找不到 {csv_name}，略過。")
            continue

        base_name = os.path.splitext(csv_name)[0]  # train / val
        out_dir = f"sorted_preprocessed_{base_name}"
        process_csv(csv_name, out_dir)

    print("\n🎉 全部處理完成！")
