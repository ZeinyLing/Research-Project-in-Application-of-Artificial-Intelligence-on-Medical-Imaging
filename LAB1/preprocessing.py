import os
import cv2
from tqdm import tqdm  # 加进度条

input_base = "datasets"
output_base = "dataset_CLAHE_512"

# 建立输出根目录
os.makedirs(output_base, exist_ok=True)

# 设置 CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 遍历所有子资料夹
for root, dirs, files in os.walk(input_base):
    # 计算对应的输出路径
    rel_path = os.path.relpath(root, input_base)
    output_dir = os.path.join(output_base, rel_path)
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(files, desc=f"Processing {rel_path}"):
        in_path = os.path.join(root, file)
        out_path = os.path.join(output_dir, file)

        # 读取影像
        img = cv2.imread(in_path)
        if img is None:
            continue

        # Resize to 512x512
        img = cv2.resize(img, (512, 512))

        # 灰阶或彩色 CLAHE
        if len(img.shape) == 2:
            img_clahe = clahe.apply(img)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        # 存档
        cv2.imwrite(out_path[:-5]+".png", img_clahe)

print("✅ 所有影像已处理完成，并存到 dataset_CLAHE_512/")
