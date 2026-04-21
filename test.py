import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入你的新版 dataset 和 model
from dataset import DroneDataset
from model import DualChannelDroneNet,DroneNet_1D_Only,DroneNet_2D_Only

def test():
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 请根据你实际训练时的类别修改这里
    class_names = ['T10000', 'T10001','T10010','T10011', 'T10100','T10101','T10110','T10111','T11000']

    # 2. 加载数据
    # 注意：T=2096 必须与训练时保持一致
    test_loader = DataLoader(DroneDataset('test.txt', N=512, T=2096), batch_size=8, shuffle=False)

    # 3. 加载模型
    # 注意：num_classes 必须与你训练时的类别数一致 (你现在好像是 3 类)
    model = DualChannelDroneNet(num_classes=9).to(device)
    
    # 请确保文件名正确
    model_path = 'best_drone_model_SEdual.pth'
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded successfully from {model_path}.")
    except FileNotFoundError:
        print(f"Error: '{model_path}' not found. Please check the file path.")
        return

    model.eval()

    y_true, y_pred = [], []

    print("\nStarting evaluation...")
    pbar = tqdm(test_loader, desc="Testing", unit="batch")

    with torch.no_grad():
        # --- 核心修改：解包双通道数据 ---
        for inputs, labels in pbar:
            # inputs 是一个列表: [img_batch, seq_batch]
            img_input, seq_input = inputs
            
            # 分别移动到 GPU
            img_input = img_input.to(device)
            seq_input = seq_input.to(device)
            
            # 传入双通道模型 (需要两个参数)
            outputs = model(img_input, seq_input)

            # 获取预测结果
            _, predicted = torch.max(outputs, 1)

            # 收集结果
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
        # -----------------------------

    # 4. 输出评估报告
    print("\n" + "=" * 50)
    print("Classification Report:")
    # digits=4 可以显示更精确的小数
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("=" * 50)

    # 5. 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title("Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    # 在服务器环境下 plt.show() 可能不显示，建议保存图片
    plt.savefig('confusion_matrix_Dual.png')
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    test()