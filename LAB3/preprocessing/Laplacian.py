import os
import cv2
import numpy as np

#--------------------------------------------------
# 直接指定資料夾 (請自行修改)
#--------------------------------------------------
input_dir  = "./train/"      
output_dir = "./train_horizontal/"    
ksize = 3                         
do_abs = True                       
normalize = True                    
#--------------------------------------------------


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def apply_laplacian(img, ksize=3, do_abs=True, normalize=True):

    # convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)

    if do_abs:
        lap = np.abs(lap)

    if normalize:
        minv, maxv = lap.min(), lap.max()
        if maxv - minv > 1e-9:
            lap_norm = 255.0 * (lap - minv) / (maxv - minv)
        else:
            lap_norm = np.zeros_like(lap)
        out = np.uint8(np.clip(lap_norm, 0, 255))
    else:
        out = np.uint8(np.clip(lap, 0, 255))

    return out


# 建立輸出資料夾
os.makedirs(output_dir, exist_ok=True)

# 處理所有影像
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

    out = apply_laplacian(img, ksize=ksize, do_abs=do_abs, normalize=normalize)

    name, _ = os.path.splitext(fname)
    out_path = os.path.join(output_dir, f"{name}_lap.png")

    success, enc = cv2.imencode(".png", out)
    if success:
        enc.tofile(out_path)
        count += 1
    else:
        print(f"無法儲存: {out_path}")

print(f"處理完成，共輸出 {count} 張影像。")
