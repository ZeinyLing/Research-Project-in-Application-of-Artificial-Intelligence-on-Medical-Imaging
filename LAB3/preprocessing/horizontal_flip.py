import os
import cv2
import numpy as np

#--------------------------------------------------
# 直接指定資料夾 (請自行修改)
#--------------------------------------------------
input_dir  = "./train/"      
output_dir = "./train_horizontal/"     
#--------------------------------------------------

def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# 水平反轉（flipCode = 1）
def horizontal_flip(img):
    return cv2.flip(img, 1)

# 建立輸出資料夾
os.makedirs(output_dir, exist_ok=True)

files = sorted(os.listdir(input_dir))
count = 0

for fname in files:
    if not is_image_file(fname):
        continue

    in_path = os.path.join(input_dir, fname)

    img = cv2.imdecode(np.fromfile(in_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"讀取失敗: {in_path}")
        continue

    # --- 水平反轉核心 ---
    flipped = horizontal_flip(img)

    name, _ = os.path.splitext(fname)
    out_path = os.path.join(output_dir, f"{name}_flipH.png")

    success, enc = cv2.imencode(".png", flipped)
    if success:
        enc.tofile(out_path)
        count += 1
    else:
        print(f"無法儲存: {out_path}")

print(f"處理完成，共輸出 {count} 張影像。")
