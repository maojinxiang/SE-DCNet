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
    raise ValueError(f'不支持的模型: {model_name}')


def train(model_name='SE_Dual'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    num_classes = infer_num_classes_from_txt('train.txt', 'val.txt', 'test.txt')
    print(f"自动识别类别数: {num_classes}")
    batch_size = 4
    epochs = 32
    base_lr = 0.0005
    num_workers = 2

    # ===========================
    # 1. 加载数据 (开启增强)
    # ===========================
    print("正在加载数据集...")
    # 训练集: train_mode=True (会随机加噪声)
    train_dataset = DroneDataset('train.txt', N=512, T=2096, train_mode=True)
    # 验证集: train_mode=False (保持干净，或者根据需要也可以设为 True)
    val_dataset = DroneDataset('val.txt', N=512, T=2096, train_mode=False)

    # 训练阶段避免最后一个 batch 仅 1 个样本，触发 BatchNorm 报错
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # ===========================
    # 2. 初始化模型
    # ===========================
    model, save_name = build_model(model_name, num_classes)
    print(f"正在初始化模型: {save_name} ...")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)

    warmup_epochs = 4
    scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epochs), eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # ===========================
    # 3. 训练循环
    # ===========================
    for epoch in range(epochs):
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] (LR={current_lr:.6f})", unit="batch")

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in pbar:
            img_input, seq_input = inputs
            img_input = img_input.to(device)
            seq_input = seq_input.to(device)
            labels = labels.to(device)

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
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)

        # --- 验证 ---
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                img_input, seq_input = inputs
                img_input = img_input.to(device)
                seq_input = seq_input.to(device)
                labels = labels.to(device)

                outputs = model(img_input, seq_input)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)

        print(
            f"-> Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% "
            f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%"
        )

        scheduler.step()

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            # 命名为 SE_Dual，方便区分
            torch.save(model.state_dict(), f'best_drone_model_{save_name}.pth')
            print(f"*** 最优模型已保存 (Acc: {acc:.2f}%) ***")

    # 绘制曲线
    plot_curves(history, epochs, save_name=save_name)
    save_history(history, save_name=save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train one ablation model.')
    parser.add_argument(
        '--model',
        default='SE_Dual',
        choices=['SE_Dual', '1D_Only', '2D_Only', 'ResNet_Only', 'TCN_Only'],
        help='选择要训练的模型',
    )
    args = parser.parse_args()
    train(model_name=args.model)