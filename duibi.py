import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. 根目录（即包含 runs_* 文件夹的目录）
# -------------------------------
ROOT = r"E:\毕业\duibi-main"

# -------------------------------
# 2. 定义两个模型的前缀
# -------------------------------
models = {
    "cnn_transformer": "runs_cnn_transformer_",
    "perturbawarenet": "runs_perturbawarenet_",
}

snrs = ["-5dB","0dB", "5dB", "10dB", "15dB" ]

results = {m: [] for m in models}

# -------------------------------
# 3. 读取各 summary.csv
# -------------------------------
for model_name, prefix in models.items():
    for snr in snrs:
        folder = os.path.join(ROOT, prefix + snr)
        summary_file = os.path.join(folder, "summary.csv")

        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file)
            test_acc = df["test_top1"].iloc[-1]   # 最新一条结果
        else:
            test_acc = None

        results[model_name].append(test_acc)

# -------------------------------
# 4. 绘制柱状图
# -------------------------------
plt.figure(figsize=(10,6))

bar_width = 0.35
x = range(len(snrs))

plt.bar([i - bar_width/2 for i in x], results["cnn_transformer"],
        width=bar_width, label="CNN Transformer")
plt.bar([i + bar_width/2 for i in x], results["perturbawarenet"],
        width=bar_width, label="PerturbAwareNet")

plt.xticks(x, snrs, fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.xlabel("SNR", fontsize=12)
plt.title("Model Comparison under Different SNR", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

out_path = os.path.join(ROOT, "model_snr_compare.png")
plt.savefig(out_path, dpi=200)
plt.show()

print("已生成图像：", out_path)
