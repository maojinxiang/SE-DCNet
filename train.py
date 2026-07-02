import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import matplotlib.pyplot as plt

# 引入更新后的 Dataset 和 Model
from dataset import DroneDataset
from model import (
    DualChannelDroneNet,
    DroneNet_1D_Only,
    DroneNet_2D_Only,
    DroneNet_ResNet_Only,
    DroneNet_TCN_Only,
    DroneNet_CNN_LSTM,
    DroneNet_CrossAttention,
    DualChannelConcatNet,
    DualChannelWeightedFusionNet,
)


def iter_text_lines_with_fallback(txt_path):
    encodings = ['utf-8', 'gbk', 'gb18030', 'latin-1']
    last_error = None
    for enc in encodings:
        try:
            with open(txt_path, 'r', encoding=enc) as f:
                for line in f:
                    yield line
            return
        except UnicodeDecodeError as e:
            last_error = e
            continue

    if last_error is None:
        raise UnicodeDecodeError('unknown', b'', 0, 1, f'无法解码文件: {txt_path}')
    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f'无法使用 {encodings} 解码文件: {txt_path}',
    )


def infer_num_classes_from_txt(*txt_paths):
    labels = set()
    for txt_path in txt_paths:
        try:
            for line in iter_text_lines_with_fallback(txt_path):
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    labels.add(int(parts[1]))
        except FileNotFoundError:
            continue
    if not labels:
        raise ValueError('No labels found in txt files. Please regenerate train/val/test txt first.')
    return max(labels) + 1

def plot_curves(history, epochs, save_name):
    # 绘制并保存 Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_name}_loss.png')
    plt.close()

    # 绘制并保存 Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Training Accuracy', color='green', linewidth=2)
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Validation Accuracy', color='red', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_name}_acc.png')
    plt.close()


def save_history(history, save_name):
    history_path = f'{save_name}_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"训练历史已保存到: {history_path}")


def build_model(model_name, num_classes):
    if model_name == 'SE_Dual':
        return DualChannelDroneNet(num_classes=num_classes), 'SE_Dual'
    if model_name == '1D_Only':
        return DroneNet_1D_Only(num_classes=num_classes), '1D_Only'
    if model_name == '2D_Only':
        return DroneNet_2D_Only(num_classes=num_classes), '2D_Only'
    if model_name == 'ResNet_Only':
        return DroneNet_ResNet_Only(num_classes=num_classes), 'ResNet_Only'
    if model_name == 'TCN_Only':
        return DroneNet_TCN_Only(num_classes=num_classes), 'TCN_Only'
    if model_name == 'CNN_LSTM':
        return DroneNet_CNN_LSTM(num_classes=num_classes), 'CNN_LSTM'
    if model_name == 'Cross_Attention':
        return DroneNet_CrossAttention(num_classes=num_classes), 'Cross_Attention'
    if model_name == 'Fusion_Concat':
        return DualChannelConcatNet(num_classes=num_classes), 'Fusion_Concat'
    if model_name == 'Fusion_Weighted':
        return DualChannelWeightedFusionNet(num_classes=num_classes), 'Fusion_Weighted'

    raise ValueError(f'不支持的模型: {model_name}')

import os
import csv
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了结果可复现
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(args, save_dir, num_classes):
    config = vars(args).copy()
    config['num_classes'] = num_classes
    config['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def append_csv_log(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    num_classes = infer_num_classes_from_txt('train.txt', 'val.txt', 'test.txt')
    print(f"自动识别类别数: {num_classes}")

    # ===========================
    # 创建保存目录
    # ===========================
    time_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_root) / f"{args.model}_{time_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"本次实验保存目录: {save_dir}")

    save_config(args, save_dir, num_classes)

    # ===========================
    # 1. 加载数据
    # ===========================
    print("正在加载数据集...")

    train_dataset = DroneDataset('train.txt', N=512, T=2096, train_mode=True)
    val_dataset = DroneDataset('val.txt', N=512, T=2096, train_mode=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ===========================
    # 2. 初始化模型
    # ===========================
    model, save_name = build_model(args.model, num_classes)
    print(f"正在初始化模型: {save_name} ...")
    model = model.to(device)

    total_params = count_parameters(model)
    print(f"Trainable Parameters: {total_params / 1e6:.3f} M")

    with open(save_dir / 'model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(str(model))
        f.write('\n\n')
        f.write(f"Trainable Parameters: {total_params / 1e6:.3f} M\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_epochs = 4
    scheduler1 = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    scheduler2 = CosineAnnealingLR(
        optimizer,
        T_max=(args.epochs - warmup_epochs),
        eta_min=0
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_epochs]
    )

    best_acc = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }

    csv_path = save_dir / 'epoch_log.csv'

    # ===========================
    # 3. 训练循环
    # ===========================
    for epoch in range(args.epochs):
        epoch_start = time.time()

        model.train()
        current_lr = optimizer.param_groups[0]['lr']

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{args.epochs}] {args.model} LR={current_lr:.6f}",
            unit="batch"
        )

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in pbar:
            img_input, seq_input = inputs
            img_input = img_input.to(device, non_blocking=True)
            seq_input = seq_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(img_input, seq_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = 100 * correct_train / total_train

        # ===========================
        # 验证
        # ===========================
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                img_input, seq_input = inputs
                img_input = img_input.to(device, non_blocking=True)
                seq_input = seq_input.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(img_input, seq_input)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        scheduler.step()

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(
            f"-> Epoch {epoch + 1} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # 每个 epoch 保存 csv
        append_csv_log(
            csv_path,
            {
                'epoch': epoch + 1,
                'lr': current_lr,
                'train_loss': avg_train_loss,
                'train_acc': avg_train_acc,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'epoch_time_sec': epoch_time
            }
        )

        # 每个 epoch 保存 history
        with open(save_dir / 'history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # 保存 last checkpoint
        last_ckpt = {
            'epoch': epoch + 1,
            'model_name': args.model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'history': history,
            'args': vars(args)
        }

        torch.save(last_ckpt, save_dir / 'checkpoint_last.pth')

        # 保存 best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc

            best_ckpt = {
                'epoch': epoch + 1,
                'model_name': args.model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'history': history,
                'args': vars(args)
            }

            torch.save(best_ckpt, save_dir / 'checkpoint_best.pth')

            # 兼容你原来 test.py 的权重读取方式
            torch.save(model.state_dict(), save_dir / f'best_drone_model_{save_name}.pth')

            print(f"*** 最优模型已保存: {best_acc:.2f}% ***")

    # 训练结束后画曲线
    plot_curves(history, args.epochs, save_name=str(save_dir / save_name))
    save_history(history, save_name=str(save_dir / save_name))

    print(f"\n训练完成。Best Val Acc: {best_acc:.2f}%")
    print(f"所有结果已保存到: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train one ablation or baseline model.')

    parser.add_argument(
        '--model',
        default='SE_Dual',
        choices=[
            'SE_Dual',
            '1D_Only',
            '2D_Only',
            'ResNet_Only',
            'TCN_Only',
            'CNN_LSTM',
            'Cross_Attention',
            'Fusion_Concat',
            'Fusion_Weighted',
        ],
        help='选择要训练的模型',
    )

    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_root', type=str, default='runs')

    args = parser.parse_args()
    train(args=args)