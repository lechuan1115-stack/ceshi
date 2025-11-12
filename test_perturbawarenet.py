#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""单独测试 PerturbAwareNet 的脚本。
- 直接运行本文件即可，无需命令行解析；将下面的路径改成你自己的即可。
- 确保模型结构与训练时完全一致，仅通过加载权重复现结果。
- 统计五种扰动的分类性能、扰动检测/回归指标，并生成相应图像。
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mydata_read
from mymodel1 import PerturbAwareNet, PERTURB_ORDER


# ======================== 手动配置区域 ========================
@dataclass
class TestConfig:
    data_path: str = r"E:\\数据集\\ADS-B_Test_-5dB.mat"      # 测试集 .mat 路径
    checkpoint_path: str = r"E:\\模型权重\\perturbawarenet_best.pt"  # 训练好的模型权重
    output_dir: str = "./perturbawarenet_test"                  # 输出目录
    batch_size: int = 128
    num_workers: int = 0
    use_cuda: bool = True
    z_threshold: float = 0.5


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_dataset(cfg: TestConfig):
    X, Y, Z, S, fs, snr_db, noise_var, order = mydata_read.load_adsb_aug5_strict(cfg.data_path, shuffle=False)
    if list(order) != list(PERTURB_ORDER):
        raise ValueError(f"扰动顺序不一致：{order} vs {PERTURB_ORDER}")
    tensors = (
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).long(),
        torch.from_numpy(Z).float(),
        torch.from_numpy(S).float(),
    )
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return loader, int(np.unique(Y).size), float(fs)


def build_model(num_classes: int, fs: float, device: torch.device, cfg: TestConfig) -> PerturbAwareNet:
    model = PerturbAwareNet(n_classes=num_classes, fs=fs)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError("checkpoint 格式不支持，请确认是 state_dict 或包含 state_dict 的字典")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"加载权重失败：missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


def evaluate(model: PerturbAwareNet, loader: DataLoader, device: torch.device, cfg: TestConfig):
    all_logits = []
    all_targets = []
    all_z_true = []
    all_s_true = []
    all_z_logit = []
    all_s_pred = []

    with torch.no_grad():
        for x, y, z, s in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            s = s.to(device)
            logits, feat, z_logit, s_pred = model(x, z, s)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
            all_z_true.append(z.cpu())
            all_s_true.append(s.cpu())
            all_z_logit.append(z_logit.cpu())
            all_s_pred.append(s_pred.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    z_true = torch.cat(all_z_true)
    s_true = torch.cat(all_s_true)
    z_logit = torch.cat(all_z_logit)
    s_pred = torch.cat(all_s_pred)

    y_prob = torch.softmax(logits, dim=1)
    y_pred = torch.argmax(y_prob, dim=1)

    overall_acc = accuracy_score(targets.numpy(), y_pred.numpy())
    overall_err = 1.0 - overall_acc

    z_prob = torch.sigmoid(z_logit)
    z_pred = (z_prob > cfg.z_threshold).float()

    perturb_metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(PERTURB_ORDER):
        mask = z_true[:, idx] > 0.5
        mask_np = mask.numpy()
        perturb_metrics[name] = {}

        if mask_np.sum() > 0:
            y_true_masked = targets[mask].numpy()
            y_pred_masked = y_pred[mask].numpy()
            acc = accuracy_score(y_true_masked, y_pred_masked)
            perturb_metrics[name]["cls_acc"] = float(acc)
            perturb_metrics[name]["cls_err"] = float(1.0 - acc)
        else:
            perturb_metrics[name]["cls_acc"] = float("nan")
            perturb_metrics[name]["cls_err"] = float("nan")

        z_true_bin = z_true[:, idx].numpy().astype(int)
        z_pred_bin = z_pred[:, idx].numpy().astype(int)
        acc_z = accuracy_score(z_true_bin, z_pred_bin)
        prec, rec, f1, _ = precision_recall_fscore_support(
            z_true_bin,
            z_pred_bin,
            average="binary",
            zero_division=0,
        )
        perturb_metrics[name]["z_acc"] = float(acc_z)
        perturb_metrics[name]["z_prec"] = float(prec)
        perturb_metrics[name]["z_rec"] = float(rec)
        perturb_metrics[name]["z_f1"] = float(f1)

        active = mask_np
        if active.any():
            s_true_active = s_true[active, idx].numpy()
            s_pred_active = s_pred[active, idx].numpy()
            mse = float(np.mean((s_true_active - s_pred_active) ** 2))
            mae = float(np.mean(np.abs(s_true_active - s_pred_active)))
            rmse = float(np.sqrt(mse))
        else:
            mse = mae = rmse = float("nan")
        perturb_metrics[name]["s_mse"] = mse
        perturb_metrics[name]["s_rmse"] = rmse
        perturb_metrics[name]["s_mae"] = mae

    results = {
        "overall_accuracy": float(overall_acc),
        "overall_error": float(overall_err),
        "perturb_metrics": perturb_metrics,
    }

    return results, targets.numpy(), y_pred.numpy(), z_true.numpy(), z_pred.numpy()


def save_metrics(results: Dict, cfg: TestConfig, out_dir: str):
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 指标已写入: {json_path}")


def plot_bar(values: List[float], labels: List[str], ylabel: str, title: str, out_path: str):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values, color="#3c8dbc")
    plt.xticks(range(len(values)), labels, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] {title} -> {out_path}")


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Classification Confusion Matrix")
    plt.colorbar()
    ticks = range(cm.shape[0])
    plt.xticks(ticks, ticks, rotation=90)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Confusion matrix -> {out_path}")


def generate_figures(results: Dict, y_true: np.ndarray, y_pred: np.ndarray, out_dir: str):
    labels = list(results["perturb_metrics"].keys())
    cls_acc = [results["perturb_metrics"][k]["cls_acc"] for k in labels]
    z_acc = [results["perturb_metrics"][k]["z_acc"] for k in labels]
    s_rmse = [results["perturb_metrics"][k]["s_rmse"] for k in labels]

    plot_bar(cls_acc, labels, "Accuracy", "分类准确率（按扰动）", os.path.join(out_dir, "cls_acc_bar.png"))
    plot_bar(z_acc, labels, "Accuracy", "扰动检测准确率", os.path.join(out_dir, "z_acc_bar.png"))
    plot_bar(s_rmse, labels, "RMSE", "扰动参数回归 RMSE", os.path.join(out_dir, "s_rmse_bar.png"))
    plot_confusion(y_true, y_pred, os.path.join(out_dir, "confusion_matrix.png"))


def main():
    cfg = TestConfig()
    device = torch.device("cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")
    out_dir = ensure_dir(cfg.output_dir)

    loader, num_classes, fs = load_dataset(cfg)
    model = build_model(num_classes=num_classes, fs=fs, device=device, cfg=cfg)
    results, y_true, y_pred, _, _ = evaluate(model, loader, device, cfg)

    print("整体分类准确率: {:.4f}".format(results["overall_accuracy"]))
    save_metrics(results, cfg, out_dir)
    generate_figures(results, y_true, y_pred, out_dir)


if __name__ == "__main__":
    main()
