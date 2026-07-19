# -*- coding: utf-8 -*-
import os
import re
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
# ========= 可自行修改 =========
DATA_DIR    = './dataset_CLAHE_512/test/'
IMG_SIZE    = 512
MODEL_PATHS = [

    "./pkls/vgg16_ep_20.pkl",
    "./pkls/resnet50_ep_20.pkl",
    "./pkls/resnet34_ep_20.pkl",
    "./pkls/resnet18_ep_20.pkl",
    # 'resnet18_ep_20.pkl',
]
USE_NORMALIZE = False
SAVE_FIGS   = True
OUT_DIR     = './cm_plots_voting'
CM_CMAP     = 'magma'
SHOW_PERCENT_IN_CELL = False
# =============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_tf_ops = [transforms.ToTensor()]
if USE_NORMALIZE:
    _tf_ops.append(transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]))
TO_TENSOR = transforms.Compose(_tf_ops)

def load_model(path):
    mdl = torch.load(path, map_location=device)
    mdl.eval()
    return mdl.to(device)

def list_images_and_labels(root):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not classes:
        raise RuntimeError(f'在 {root} 找不到類別子資料夾。')
    samples = []
    for ci, cname in enumerate(classes):
        cdir = os.path.join(root, cname)
        for fn in os.listdir(cdir):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                samples.append((os.path.join(cdir, fn), ci))
    if not samples:
        raise RuntimeError(f'在 {root} 找不到任何影像檔。')
    return samples, classes

def resize_to_rgb(img_bgr, size):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[:2] != (size, size):
        img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return img_rgb

def predict_single(model, img_tensor):
    with torch.no_grad():
        logits = model(img_tensor)             # [1, C]
        pred = torch.argmax(logits, dim=1)     # [1]
    return int(pred.item())

def plot_cm(cm, classes, title='Confusion Matrix', normalize=False, save_path=None, cmap='YlGnBu'):
    cm = np.array(cm, dtype=np.float64)
    disp_vals = cm.copy()
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
        disp_vals = cm / row_sums

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(disp_vals, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label',
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = disp_vals.max() / 2.0 if disp_vals.size else 0.5
    for i in range(disp_vals.shape[0]):
        for j in range(disp_vals.shape[1]):
            if normalize or SHOW_PERCENT_IN_CELL:
                txt = f"{cm[i, j]:.0f}\n({disp_vals[i, j]*100:.1f}%)"
            else:
                txt = f"{cm[i, j]:.0f}"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    color="white" if disp_vals[i, j] < thresh else "black",
                    fontsize=10)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.show()

def sanitize_filename(s):
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', s)

def main():
    t0 = time.time()
    samples, class_names = list_images_and_labels(DATA_DIR)
    num_classes = len(class_names)

    # 載入多個模型
    models = [load_model(p) for p in MODEL_PATHS]

    # 各模型單獨的混淆矩陣
    cms = [np.zeros((num_classes, num_classes), dtype=np.int64) for _ in models]
    # Voting 混淆矩陣
    cm_voted = np.zeros((num_classes, num_classes), dtype=np.int64)

    # 儲存真實與預測（方便 sklearn 計算）
    y_true_all = []
    y_pred_voted_all = []
    y_preds_models = [[] for _ in models]

    for img_path, true_idx in samples:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[警告] 讀取失敗，略過：{img_path}")
            continue
        rgb = resize_to_rgb(bgr, IMG_SIZE)
        tensor = TO_TENSOR(rgb).unsqueeze(0).to(device)

        preds = []
        for mi, mdl in enumerate(models):
            pred = predict_single(mdl, tensor)
            preds.append(pred)
            cms[mi][true_idx, pred] += 1
            y_preds_models[mi].append(pred)

        final_pred = np.bincount(preds).argmax()
        cm_voted[true_idx, final_pred] += 1

        y_true_all.append(true_idx)
        y_pred_voted_all.append(final_pred)

    if SAVE_FIGS:
        os.makedirs(OUT_DIR, exist_ok=True)

    print(f"共測試影像：{len(samples)}")

    # 各模型結果
    for i, (path, cm, preds_i) in enumerate(zip(MODEL_PATHS, cms, y_preds_models), start=1):
        y_true_np = np.array(y_true_all)
        y_pred_np = np.array(preds_i)
        acc = np.trace(cm) / (cm.sum() + 1e-12)
        f1 = f1_score(y_true_np, y_pred_np, average='macro')
        title = f"Model {i}: {os.path.basename(path)} | Acc={acc*100:.2f}% | F1={f1*100:.2f}%"
        print(f"[{i}] {path}\nConfusion Matrix:\n{cm}\nAcc: {acc:.4f}, F1: {f1:.4f}\n")
        save_path = os.path.join(OUT_DIR, f"cm_{i}_{sanitize_filename(os.path.basename(path))}.png") if SAVE_FIGS else None
        plot_cm(cm, class_names, title=title, normalize=False, save_path=save_path, cmap=CM_CMAP)

    # Voting 結果
    acc_voted = np.trace(cm_voted) / (cm_voted.sum() + 1e-12)
    f1_voted = f1_score(y_true_all, y_pred_voted_all, average='macro')
    title_voted = f"Voting Ensemble | Acc={acc_voted*100:.2f}% | F1={f1_voted*100:.2f}%"

    print(f"[Voting]\nConfusion Matrix:\n{cm_voted}\nAcc: {acc_voted:.4f}, F1: {f1_voted:.4f}\n")
    save_path = os.path.join(OUT_DIR, "cm_voted.png") if SAVE_FIGS else None
    plot_cm(cm_voted, class_names, title=title_voted, normalize=False, save_path=save_path, cmap=CM_CMAP)

    print(f"總耗時：{time.time() - t0:.2f} 秒")

if __name__ == "__main__":
    main()
