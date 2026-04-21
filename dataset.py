import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


def compute_stft_torch(i_signal, q_signal, nperseg, noverlap, nfft):
    """
    使用 torch.stft 计算复信号 STFT，避免 scipy 依赖导致的导入问题。
    返回值形状与原逻辑一致: (freq_bins, time_bins), 复数矩阵。
    """
    hop_length = max(1, nperseg - noverlap)

    i_tensor = torch.from_numpy(np.asarray(i_signal, dtype=np.float32))
    q_tensor = torch.from_numpy(np.asarray(q_signal, dtype=np.float32))
    window = torch.hann_window(nperseg)

    stft_i = torch.stft(
        i_tensor,
        n_fft=nfft,
        hop_length=hop_length,
        win_length=nperseg,
        window=window,
        center=True,
        onesided=False,
        return_complex=True,
    )
    stft_q = torch.stft(
        q_tensor,
        n_fft=nfft,
        hop_length=hop_length,
        win_length=nperseg,
        window=window,
        center=True,
        onesided=False,
        return_complex=True,
    )

    zxx = stft_i + 1j * stft_q
    return zxx.cpu().numpy()

def add_random_noise(signal, min_snr=0, max_snr=20):
    """
    随机添加噪声 (训练增强专用)
    :param min_snr: 最小信噪比
    :param max_snr: 最大信噪比
    """
    # 随机选择一个 SNR
    target_snr = np.random.uniform(min_snr, max_snr)
    
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (target_snr / 10.0)
    
    # 避免除以0
    if snr_linear == 0: snr_linear = 1e-9
    
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    
    return signal + noise

class DroneDataset(Dataset):
    def __init__(self, txt_path, N=512, T=2096, train_mode=False):
        """
        Args:
            train_mode (bool): 是否为训练模式。如果是 True，会随机加噪声进行增强。
        """
        with open(txt_path, 'r') as f:
            self.samples = [line.strip().split(',') for line in f.readlines()]
            
        self.N = N
        self.N_overlap = N // 2
        self.T = T
        self.train_mode = train_mode # 记录模式
        
        self.len_1d = 1000000 
        self.len_2d = N + (T - 1) * self.N_overlap
        self.read_len = max(self.len_1d, self.len_2d)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, offset = self.samples[idx]
        offset = int(offset)

        try:
            with h5py.File(file_path, 'r', libver='latest', swmr=True) as f:
                # 兼容不同维度的存储格式
                if 'RF0_I' in f:
                    i_ds, q_ds = f['RF0_I'], f['RF0_Q']
                else:
                    keys = list(f.keys())
                    i_ds, q_ds = f[keys[0]], f[keys[1]]

                shape = i_ds.shape
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

        if len(i_data) < self.read_len:
            pad_len = self.read_len - len(i_data)
            i_data = np.pad(i_data, (0, pad_len), 'constant')
            q_data = np.pad(q_data, (0, pad_len), 'constant')
        elif len(i_data) > self.read_len:
            i_data = i_data[:self.read_len]
            q_data = q_data[:self.read_len]

        # ==========================================
        # 【关键修改】训练时随机加噪声！
        # ==========================================
        if self.train_mode:
            # 50% 的概率加噪声，50% 保持干净
            # 这样模型既见过恶劣环境(-5dB)，也见过良好环境(15dB)
            if np.random.rand() > 0.5: 
                i_data = add_random_noise(i_data, min_snr=-5, max_snr=15)
                q_data = add_random_noise(q_data, min_snr=-5, max_snr=15)

        # --- 分支 A: 1D ---
        i_1d = i_data[:self.len_1d]
        q_1d = q_data[:self.len_1d]
        seq_feature = np.stack([i_1d, q_1d], axis=0)

        # --- 分支 B: 2D ---
        i_2d = i_data[:self.len_2d]
        q_2d = q_data[:self.len_2d]
        
        Zxx = compute_stft_torch(
            i_signal=i_2d,
            q_signal=q_2d,
            nperseg=self.N,
            noverlap=self.N_overlap,
            nfft=self.N,
        )
        
        if Zxx.shape[1] > self.T:
            Zxx = Zxx[:, :self.T]
        elif Zxx.shape[1] < self.T:
            pad_w = self.T - Zxx.shape[1]
            Zxx = np.pad(Zxx, ((0,0), (0, pad_w)), 'constant')

        real = np.real(Zxx)
        imag = np.imag(Zxx)
        real_norm = (real - np.mean(real)) / (np.std(real) + 1e-7)
        imag_norm = (imag - np.mean(imag)) / (np.std(imag) + 1e-7)

        stft_feature = np.stack([real_norm, imag_norm], axis=0)

        return (torch.from_numpy(stft_feature).float(), torch.from_numpy(seq_feature).float()), int(label)