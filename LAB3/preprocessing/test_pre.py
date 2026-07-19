# -*- coding: utf-8 -*-
"""
影像前處理 (Test 無標籤版本)
---------------------------------
輸入：
- images_test/
輸出：
- preprocessed_test/
"""

import os
import cv2
from tqdm import tqdm

# ======== 路徑設定 ========
IMG_DIR  = "test_images"         # 原始測試影像資料夾
OUT_DIR  = "test"   # 前處理後的輸出資料夾
IMG_SIZE = (512, 512)            # resize 尺寸
APPLY_CLAHE = True               # 是否使用 CLAHE
NORMALIZE = True                 # 是否標準化

# ======== CLAHE 函數 ========
def apply_clahe_color(img):
    """對彩色影像進行 CLAHE 增強"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ======== 主流程 ========
def preprocess_images(img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_list = [f for f in os.listdir(img_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"📂 找到 {len(img_list)} 張測試影像")
    for fname in tqdm(img_list):
        src_path = os.path.join(img_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            print(f"[Error] 無法讀取: {fname}")
            continue

        # Resize
        img = cv2.resize(img, IMG_SIZE)

        # CLAHE
        if APPLY_CLAHE:
            img = apply_clahe_color(img)

        # Normalize
        if NORMALIZE:
            img = (img / 255.0)
            img = (img * 255).astype("uint8")

        dst_path = os.path.join(out_dir, fname)
        cv2.imwrite(dst_path, img)

    print(f"\n✅ 前處理完成！共處理 {len(img_list)} 張影像")
    print(f"📁 儲存至：{out_dir}")

# ======== 執行入口 ========
if __name__ == "__main__":
    if not os.path.exists(IMG_DIR):
        print(f"⚠️ 找不到資料夾：{IMG_DIR}")
    else:
        preprocess_images(IMG_DIR, OUT_DIR)
