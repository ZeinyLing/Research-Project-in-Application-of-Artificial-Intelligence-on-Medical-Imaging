import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

# ========= 使用者設定 =========
input_dir = "csvs"      # 輸入 CSV 資料夾
output_dir = "plots"    # 圖片輸出資料夾
os.makedirs(output_dir, exist_ok=True)

metrics = ["accuracy", "f1"]
phases = ["train", "val"]

# ========= 搜尋所有 CSV =========
csv_files = sorted(glob(os.path.join(input_dir, "*.csv")))

if not csv_files:
    print("⚠️ 沒有找到 CSV 檔案，請確認路徑是否正確。")
else:
    for csv_path in csv_files:
        exp_name = os.path.splitext(os.path.basename(csv_path))[0]
        save_dir = os.path.join(output_dir, exp_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"📊 處理檔案: {exp_name}")

        # 讀取 CSV
        df = pd.read_csv(csv_path)

        # 逐個 metric + phase 畫圖並儲存
        for metric in metrics:
            for phase in phases:
                phase_df = df[df["phase"] == phase]

                plt.figure(figsize=(6, 4))
                plt.plot(phase_df["epoch"], phase_df[metric], linewidth=2)
                plt.xlabel("Epoch")
                plt.ylabel(metric.capitalize())
                plt.title(f"{phase.capitalize()} {metric.capitalize()} - {exp_name.split("_")[0]+"_epoch_"+exp_name.split("_")[-1]}")                

                # X 軸每 5 epoch 一個刻度
                epochs = sorted(phase_df["epoch"].unique())
                plt.xticks(range(min(epochs), max(epochs) + 1, 5))

                plt.tight_layout()
                plt.grid(False)  # 移除格線

                # 儲存檔案
                save_path = os.path.join(save_dir, f"{phase}_{metric}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()

        print(f"✅ 已輸出圖檔至: {save_dir}\n")

print("🎯 所有 CSV 檔案皆已完成繪圖。")





