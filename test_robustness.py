import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
from scipy.signal import stft
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# ==========================================
# 0. 引入你的项目文件
# ==========================================
# 确保 dataset_new.py 和 model.py 在同一目录下
from model import (
    DualChannelDroneNet,
    DroneNet_1D_Only,
    DroneNet_2D_Only,
    DroneNet_ResNet_Only,
    DroneNet_TCN_Only,
)
from dataset_new import DroneDataset

# ==========================================
# 1. 定义支持在线加噪的 Dataset
# ==========================================
def add_awgn(signal, target_snr_db):
    """
    给信号添加高斯白噪声
    :param signal: 输入信号 (numpy array)
    :param target_snr_db: 目标信噪比 (dB)
    """
    # 计算信号功率
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # 转换 SNR dB 为线性值
    snr_linear = 10 ** (target_snr_db / 10.0)
    
    # 计算噪声功率
    noise_power = signal_power / snr_linear
    
    # 生成噪声 (标准差 = sqrt(noise_power))
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    
    return signal + noise

class RobustDroneDataset(DroneDataset):
    def __init__(self, txt_path, N=512, T=2096, snr_db=None):
        # 继承父类的初始化
        super().__init__(txt_path, N, T)
        self.snr_db = snr_db  # 当前测试的 SNR

    def set_snr(self, snr_db):
        """动态设置 SNR"""
        self.snr_db = snr_db

    def __getitem__(self, idx):
        # --- 1. 读取数据 (复制父类逻辑) ---
        file_path, label, offset = self.samples[idx]
        offset = int(offset)

        try:
            with h5py.File(file_path, 'r', libver='latest', swmr=True) as f:
                if 'RF0_I' in f:
                    i_ds, q_ds = f['RF0_I'], f['RF0_Q']
                else:
                    keys = list(f.keys())
                    i_ds, q_ds = f[keys[0]], f[keys[1]]

                shape = i_ds.shape
                # 处理各种 shape 情况
                if len(shape) == 2 and shape[0] == 1:
                    total_len = shape[1]
                    end_pos = min(offset + self.read_len, total_len)
                    i_data = i_ds[0, offset:end_pos]
                    q_data = q_ds[0, offset:end_pos]
                elif len(shape) == 2 and shape[1] == 1:
                    total_len = shape[0]
                    end_pos = min(offset + self.read_len, total_len)
                    i_data = i_ds[offset:end_pos, 0]
                    q_data = q_ds[offset:end_pos, 0]
                else:
                    total_len = shape[0]
                    end_pos = min(offset + self.read_len, total_len)
                    i_data = i_ds[offset:end_pos]
                    q_data = q_ds[offset:end_pos]

        except Exception as e:
            i_data = np.zeros(self.read_len)
            q_data = np.zeros(self.read_len)

        i_data = i_data.astype(np.float32)
        q_data = q_data.astype(np.float32)

        # Padding
        if len(i_data) < self.read_len:
            pad_len = self.read_len - len(i_data)
            i_data = np.pad(i_data, (0, pad_len), 'constant')
            q_data = np.pad(q_data, (0, pad_len), 'constant')
        elif len(i_data) > self.read_len:
            i_data = i_data[:self.read_len]
            q_data = q_data[:self.read_len]

        # --- 2. 【核心】在线加噪 ---
        # 必须在 STFT 变换之前加噪声，这样才是物理上正确的
        if self.snr_db is not None:
            i_data = add_awgn(i_data, self.snr_db)
            q_data = add_awgn(q_data, self.snr_db)

        # --- 3. 生成 1D 分支数据 ---
        i_1d = i_data[:self.len_1d]
        q_1d = q_data[:self.len_1d]
        seq_feature = np.stack([i_1d, q_1d], axis=0)

        # --- 4. 生成 2D 分支数据 (STFT) ---
        i_2d = i_data[:self.len_2d]
        q_2d = q_data[:self.len_2d]
        
        sig = i_2d + 1j * q_2d
        _, _, Zxx = stft(sig, nperseg=self.N, noverlap=self.N_overlap, nfft=self.N, return_onesided=False)
        
        # 尺寸微调
        if Zxx.shape[1] > self.T:
            Zxx = Zxx[:, :self.T]
        elif Zxx.shape[1] < self.T:
            pad_w = self.T - Zxx.shape[1]
            Zxx = np.pad(Zxx, ((0,0), (0, pad_w)), 'constant')

        real = np.real(Zxx)
        imag = np.imag(Zxx)
        
        # 归一化 (Z-Score)
        real_norm = (real - np.mean(real)) / (np.std(real) + 1e-7)
        imag_norm = (imag - np.mean(imag)) / (np.std(imag) + 1e-7)

        stft_feature = np.stack([real_norm, imag_norm], axis=0)

        return (torch.from_numpy(stft_feature).float(), torch.from_numpy(seq_feature).float()), int(label)

# ==========================================
# 2. 带有进度条的评估函数
# ==========================================
def evaluate_model(model, dataloader, device, model_type='dual', desc="Eval"):
    model.eval()
    correct = 0
    total = 0
    
    # --- 进度条配置 ---
    # desc: 显示当前正在测哪个模型和 SNR
    # leave=False: 跑完一个进度条就清除，防止控制台刷屏
    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=False, ncols=100)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            img_input, seq_input = inputs
            img_input, seq_input = img_input.to(device), seq_input.to(device)
            labels = labels.to(device)
            
            if model_type == 'dual':
                outputs = model(img_input, seq_input)
            elif model_type == '1d':
                outputs = model(img_input, seq_input)
            elif model_type == '2d':
                outputs = model(img_input, seq_input)
                
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # (可选) 实时显示当前 batch 的准确率
            # current_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            # pbar.set_postfix({"Batch Acc": f"{current_acc:.1f}%"})
            
    return 100 * correct / total

# ==========================================
# 3. 主程序
# ==========================================
def run_robustness_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 实验配置 ---
    # SNR 测试点：从 -10dB 到 10dB
    snr_list = [-10, -5, 0, 5, 10]
    
    # 数据集加载 (使用 test.txt)
    print("Loading Test Dataset...")
    dataset = RobustDroneDataset('test.txt', N=512, T=2096, snr_db=None)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2) 
    
    # --- 模型权重路径 (请根据实际情况修改!) ---
    paths = {
        '1D-Only': 'best_drone_model_1D_Only.pth',
        '2D-Only': 'best_drone_model_2D_Only.pth',
        'ResNet-Only': 'best_drone_model_ResNet_Only.pth',
        'TCN-Only': 'best_drone_model_TCN_Only.pth',
        'SE-Dual (Proposed)': 'best_drone_model_SEdual.pth'
    }
    
    models = {}
    
    
    print("\nLoading Models...")
    # 1D 模型加载
    if os.path.exists(paths['1D-Only']):
        net1d = DroneNet_1D_Only(num_classes=9).to(device)
        net1d.load_state_dict(torch.load(paths['1D-Only'], map_location=device))
        models['1D-Only'] = net1d
        print(f"[OK] Loaded 1D-Only from {paths['1D-Only']}")
    else:
        print(f"[Warning] File not found: {paths['1D-Only']} (Skipping 1D model)")

    # 2D 模型加载
    if os.path.exists(paths['2D-Only']):
        net2d = DroneNet_2D_Only(num_classes=9).to(device)
        net2d.load_state_dict(torch.load(paths['2D-Only'], map_location=device))
        models['2D-Only'] = net2d
        print(f"[OK] Loaded 2D-Only from {paths['2D-Only']}")
    else:
        print(f"[Warning] File not found: {paths['2D-Only']} (Skipping 2D model)")

    # ResNet 对照模型加载
    if os.path.exists(paths['ResNet-Only']):
        net_resnet = DroneNet_ResNet_Only(num_classes=9).to(device)
        net_resnet.load_state_dict(torch.load(paths['ResNet-Only'], map_location=device))
        models['ResNet-Only'] = net_resnet
        print(f"[OK] Loaded ResNet-Only from {paths['ResNet-Only']}")
    else:
        print(f"[Warning] File not found: {paths['ResNet-Only']} (Skipping ResNet model)")

    # TCN 对照模型加载
    if os.path.exists(paths['TCN-Only']):
        net_tcn = DroneNet_TCN_Only(num_classes=9).to(device)
        net_tcn.load_state_dict(torch.load(paths['TCN-Only'], map_location=device))
        models['TCN-Only'] = net_tcn
        print(f"[OK] Loaded TCN-Only from {paths['TCN-Only']}")
    else:
        print(f"[Warning] File not found: {paths['TCN-Only']} (Skipping TCN model)")

    # 双通道模型加载
    if os.path.exists(paths['SE-Dual (Proposed)']):
        net_dual = DualChannelDroneNet(num_classes=9).to(device)
        net_dual.load_state_dict(torch.load(paths['SE-Dual (Proposed)'], map_location=device))
        models['SE-Dual (Proposed)'] = net_dual
        print(f"[OK] Loaded SE-Dual (Proposed) from {paths['SE-Dual (Proposed)']}")
    else:
        print(f"\033[91m[Error] File not found: {paths['SE-Dual (Proposed)']}\033[0m")
        print(f"\033[91m -> Please make sure you have trained the Dual model and saved it as '{paths['SE-Dual (Proposed)']}'\033[0m")

    if not models:
        print("\nNo models loaded. Exiting.")
        return

    # 结果保存
    results = {name: [] for name in models.keys()}

    # --- 开始测试循环 ---
    print("\n" + "="*50)
    print("Starting SNR Robustness Test (Noise Injection)")
    print("="*50)
    
    # 外层进度条 (遍历 SNR 列表)
    total_steps = len(snr_list) * len(models)
    
    for snr in snr_list:
        # 设置当前 SNR
        dataset.set_snr(snr)
        
        # 遍历所有模型
        for name, model in models.items():
            model_type = 'dual'
            if '1D' in name:
                model_type = '1d'
            elif '2D' in name or 'ResNet' in name:
                model_type = '2d'
            
            # 这里的 desc 会显示在进度条左侧
            desc_text = f"[{name}] SNR={snr}dB"
            
            acc = evaluate_model(model, dataloader, device, model_type, desc=desc_text)
            results[name].append(acc)
            
            # 使用 tqdm.write 打印，避免打断进度条
            tqdm.write(f"Model: {name:<15} | SNR: {snr:>3}dB | Accuracy: {acc:.2f}%")

    # --- 绘图 ---
    print("\nPlotting results...")
    plt.figure(figsize=(10, 6))
    
    plot_order = ['TCN-Only', '1D-Only', 'ResNet-Only', '2D-Only', 'SE-Dual (Proposed)']
    legend_names = {
        'TCN-Only': 'IQ-TCN',
        '1D-Only': 'IQ-CNN',
        'ResNet-Only': 'STFT-ResNet',
        '2D-Only': 'STFT-EfficientNet',
        'SE-Dual (Proposed)': 'Proposed SE-DCNet',
    }
    markers = {
        '1D-Only': 's',
        '2D-Only': '^',
        'ResNet-Only': 'D',
        'TCN-Only': 'P',
        'SE-Dual (Proposed)': 'o',
    }
    colors = {
        '1D-Only': '#1f77b4',
        '2D-Only': '#2ca02c',
        'ResNet-Only': '#d62728',
        'TCN-Only': '#17becf',
        'SE-Dual (Proposed)': '#ff7f0e',
    }

    for name in plot_order:
        acc_list = results.get(name, [])
        if not acc_list: continue
        plt.plot(snr_list, acc_list, marker=markers.get(name, 'o'),
                 label=legend_names.get(name, name), color=colors.get(name, 'black'), linewidth=2)
    
    plt.title('Model Robustness under Different SNR Levels', fontsize=14)
    plt.xlabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    # 图例放在图内上方居中，减少对曲线区域的遮挡
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        fontsize=10,
        frameon=True,
        framealpha=0.92,
        facecolor='white',
        edgecolor='gray',
        ncol=3,
    )
    plt.xticks(snr_list)
    plt.ylim(0, 105)
    
    save_path = 'robustness_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[Done] Plot saved to: {save_path}")

    # --- 保存 JSON 结果 ---
    json_path = 'robustness_results.json'
    json_data = {
        'snr_db': snr_list,
        'accuracy_by_model': {
            name: [round(float(acc), 4) for acc in acc_list]
            for name, acc_list in results.items()
        },
        'rows': [
            {
                'model': name,
                'snr_accuracy': {
                    str(snr): round(float(acc), 4)
                    for snr, acc in zip(snr_list, acc_list)
                },
            }
            for name, acc_list in results.items()
        ],
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"[Done] JSON saved to: {json_path}")
    
    # --- 打印 LaTeX 表格数据 (方便写论文) ---
    print("\n=== LaTeX Table Data ===")
    print("SNR (dB) & " + " & ".join([str(s) for s in snr_list]) + " \\\\")
    print("\\hline")
    for name, acc_list in results.items():
        row_str = " & ".join([f"{acc:.2f}" for acc in acc_list])
        print(f"{name} & {row_str} \\\\")

if __name__ == "__main__":
    run_robustness_test()