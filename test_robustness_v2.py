import os
import json
import csv
import glob
import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DroneDataset, compute_stft_torch
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


# ============================================================
# 1. Model registry
# ============================================================
MODEL_REGISTRY = {
    "SE_Dual": DualChannelDroneNet,
    "1D_Only": DroneNet_1D_Only,
    "2D_Only": DroneNet_2D_Only,
    "ResNet_Only": DroneNet_ResNet_Only,
    "TCN_Only": DroneNet_TCN_Only,
    "CNN_LSTM": DroneNet_CNN_LSTM,
    "Cross_Attention": DroneNet_CrossAttention,
    "Fusion_Concat": DualChannelConcatNet,
    "Fusion_Weighted": DualChannelWeightedFusionNet,
}

PLOT_LABELS = {
    "1D_Only": "IQ-CNN",
    "2D_Only": "STFT-EfficientNet",
    "ResNet_Only": "STFT-ResNet",
    "TCN_Only": "IQ-TCN",
    "CNN_LSTM": "CNN-LSTM",
    "Cross_Attention": "Cross-Attention",
    "Fusion_Concat": "Concat Fusion",
    "Fusion_Weighted": "Weighted Fusion",
    "SE_Dual": "SE-DCNet",
}

DEFAULT_MODELS = [
    "1D_Only",
    "2D_Only",
    "CNN_LSTM",
    "Cross_Attention",
    "SE_Dual",
]


# ============================================================
# 2. Utilities
# ============================================================
def parse_csv_list(text, cast=str):
    if text is None or text == "":
        return []
    return [cast(x.strip()) for x in text.split(",") if x.strip() != ""]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_channel_code(channel_type):
    return sum(ord(c) for c in channel_type)


def load_weight_mapping(weights_json):
    if weights_json is None:
        return {}
    with open(weights_json, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_weight(model_name, weight_root=".", weight_mapping=None):
    """
    Priority:
    1) explicit path in weights_json
    2) recursive search under weight_root for best_drone_model_{model_name}.pth
    3) current directory fallback
    """
    if weight_mapping and model_name in weight_mapping:
        path = weight_mapping[model_name]
        if os.path.exists(path):
            return path
        print(f"[Warning] Explicit weight path not found for {model_name}: {path}")

    patterns = [
        os.path.join(weight_root, "**", f"best_drone_model_{model_name}.pth"),
        os.path.join(weight_root, "**", "checkpoint_best.pth"),
        f"best_drone_model_{model_name}.pth",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    # For checkpoint_best.pth, only keep paths whose parent folder name starts with model name.
    filtered = []
    for p in candidates:
        base = os.path.basename(p)
        parent = os.path.basename(os.path.dirname(p))
        if base == f"best_drone_model_{model_name}.pth":
            filtered.append(p)
        elif base == "checkpoint_best.pth" and parent.startswith(model_name + "_"):
            filtered.append(p)

    if not filtered:
        return None

    filtered = sorted(filtered, key=lambda x: os.path.getmtime(x), reverse=True)
    return filtered[0]


def build_model(model_name, num_classes):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](num_classes=num_classes)


def load_weights(model, weight_path, device):
    ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Compatible with DataParallel checkpoints.
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    return model


# ============================================================
# 3. Channel models
# ============================================================
def add_awgn_complex(sig, snr_db, rng):
    signal_power = np.mean(np.abs(sig) ** 2)
    if signal_power <= 1e-12:
        return sig

    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-12)

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)
    )
    return sig + noise


def rayleigh_coeff(shape, rng, fading_mode="flat"):
    if fading_mode == "flat":
        return np.sqrt(1 / 2) * (
            rng.standard_normal() + 1j * rng.standard_normal()
        )
    if fading_mode == "fast":
        return np.sqrt(1 / 2) * (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        )
    raise ValueError(f"Unsupported fading_mode: {fading_mode}")


def rician_coeff(shape, rng, k_factor=5.0, fading_mode="flat"):
    los = np.sqrt(k_factor / (k_factor + 1.0))

    if fading_mode == "flat":
        nlos = np.sqrt(1.0 / (k_factor + 1.0)) * np.sqrt(1 / 2) * (
            rng.standard_normal() + 1j * rng.standard_normal()
        )
    elif fading_mode == "fast":
        nlos = np.sqrt(1.0 / (k_factor + 1.0)) * np.sqrt(1 / 2) * (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        )
    else:
        raise ValueError(f"Unsupported fading_mode: {fading_mode}")

    return los + nlos


def apply_channel(i_data, q_data, channel_type, snr_db, rng, k_factor=5.0, fading_mode="flat"):
    """
    Apply channel impairment to complex baseband signal before STFT.
    Returned I/Q arrays preserve the original shape.
    """
    sig = i_data.astype(np.float32) + 1j * q_data.astype(np.float32)

    if channel_type == "clean":
        out = sig
    elif channel_type == "awgn":
        out = add_awgn_complex(sig, snr_db, rng)
    elif channel_type == "rayleigh":
        h = rayleigh_coeff(sig.shape, rng, fading_mode=fading_mode)
        out = add_awgn_complex(h * sig, snr_db, rng)
    elif channel_type == "rician":
        h = rician_coeff(sig.shape, rng, k_factor=k_factor, fading_mode=fading_mode)
        out = add_awgn_complex(h * sig, snr_db, rng)
    else:
        raise ValueError(f"Unsupported channel_type: {channel_type}")

    return np.real(out).astype(np.float32), np.imag(out).astype(np.float32)


# ============================================================
# 4. Robust dataset
# ============================================================
class RobustDroneDataset(DroneDataset):
    """
    Same output format as DroneDataset:
        ((stft_feature, seq_feature), label)
    but applies AWGN/Rayleigh/Rician before generating 1D and STFT inputs.

    The channel realization is deterministic with respect to:
        sample index + channel type + SNR + base seed
    so every model is evaluated on the same impaired test samples.
    """
    def __init__(
        self,
        txt_path,
        N=512,
        T=2096,
        channel_type="awgn",
        snr_db=None,
        k_factor=5.0,
        fading_mode="flat",
        base_seed=42,
    ):
        super().__init__(txt_path, N=N, T=T, train_mode=False)
        self.channel_type = channel_type
        self.snr_db = snr_db
        self.k_factor = k_factor
        self.fading_mode = fading_mode
        self.base_seed = base_seed

    def set_channel(self, channel_type):
        self.channel_type = channel_type

    def set_snr(self, snr_db):
        self.snr_db = snr_db

    def _rng_for_sample(self, idx):
        snr_part = 0 if self.snr_db is None else int(round((float(self.snr_db) + 100.0) * 100))
        channel_part = stable_channel_code(self.channel_type)
        seed = self.base_seed + idx * 10007 + snr_part * 101 + channel_part * 1000003
        return np.random.default_rng(seed)

    def __getitem__(self, idx):
        file_path, label, offset = self.samples[idx]
        offset = int(offset)

        try:
            with h5py.File(file_path, "r", libver="latest", swmr=True) as f:
                if "RF0_I" in f:
                    i_ds, q_ds = f["RF0_I"], f["RF0_Q"]
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

        except Exception:
            i_data = np.zeros(self.read_len, dtype=np.float32)
            q_data = np.zeros(self.read_len, dtype=np.float32)

        i_data = np.asarray(i_data, dtype=np.float32)
        q_data = np.asarray(q_data, dtype=np.float32)

        if len(i_data) < self.read_len:
            pad_len = self.read_len - len(i_data)
            i_data = np.pad(i_data, (0, pad_len), "constant")
            q_data = np.pad(q_data, (0, pad_len), "constant")
        elif len(i_data) > self.read_len:
            i_data = i_data[:self.read_len]
            q_data = q_data[:self.read_len]

        # Apply channel before both 1D and STFT representations.
        if self.snr_db is not None and self.channel_type != "clean":
            rng = self._rng_for_sample(idx)
            i_data, q_data = apply_channel(
                i_data=i_data,
                q_data=q_data,
                channel_type=self.channel_type,
                snr_db=float(self.snr_db),
                rng=rng,
                k_factor=self.k_factor,
                fading_mode=self.fading_mode,
            )

        # 1D branch input
        i_1d = i_data[:self.len_1d]
        q_1d = q_data[:self.len_1d]
        seq_feature = np.stack([i_1d, q_1d], axis=0)

        # 2D branch input
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
            Zxx = np.pad(Zxx, ((0, 0), (0, pad_w)), "constant")

        real = np.real(Zxx)
        imag = np.imag(Zxx)
        real_norm = (real - np.mean(real)) / (np.std(real) + 1e-7)
        imag_norm = (imag - np.mean(imag)) / (np.std(imag) + 1e-7)

        stft_feature = np.stack([real_norm, imag_norm], axis=0)

        return (torch.from_numpy(stft_feature).float(), torch.from_numpy(seq_feature).float()), int(label)


# ============================================================
# 5. Evaluation and saving
# ============================================================
def evaluate_accuracy(model, dataloader, device, desc="Eval"):
    model.eval()
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=False, ncols=110)

    with torch.no_grad():
        for inputs, labels in pbar:
            img_input, seq_input = inputs
            img_input = img_input.to(device, non_blocking=True)
            seq_input = seq_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(img_input, seq_input)
            pred = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return 100.0 * correct / max(total, 1)


def save_csv(results, snr_list, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "model", "snr_db", "accuracy"])
        for channel, channel_data in results.items():
            for model_name, acc_list in channel_data.items():
                for snr, acc in zip(snr_list, acc_list):
                    writer.writerow([channel, model_name, snr, f"{acc:.4f}"])


def save_latex_tables(results, snr_list, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for channel, channel_data in results.items():
            f.write(f"% Channel: {channel}\n")
            f.write("\\begin{tabular}{l" + "c" * len(snr_list) + "}\n")
            f.write("\\toprule\n")
            f.write("Model & " + " & ".join([f"{snr} dB" for snr in snr_list]) + " \\\\\n")
            f.write("\\midrule\n")
            for model_name, acc_list in channel_data.items():
                label = PLOT_LABELS.get(model_name, model_name)
                row = " & ".join([f"{acc:.2f}" for acc in acc_list])
                f.write(f"{label} & {row} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n\n")


def plot_channel_curves(results, snr_list, save_dir):
    markers = ["o", "s", "^", "D", "P", "v", "*", "X"]

    for channel, channel_data in results.items():
        plt.figure(figsize=(8, 5.5))

        for idx, (model_name, acc_list) in enumerate(channel_data.items()):
            label = PLOT_LABELS.get(model_name, model_name)
            plt.plot(
                snr_list,
                acc_list,
                marker=markers[idx % len(markers)],
                linewidth=2,
                label=label,
            )

        plt.title(f"Robustness under {channel.upper()} channel")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Accuracy (%)")
        plt.xticks(snr_list)
        plt.ylim(0, 105)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()

        fig_path = os.path.join(save_dir, f"robustness_{channel}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()


# ============================================================
# 6. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation under AWGN/Rayleigh/Rician channels.")

    parser.add_argument("--test_txt", type=str, default="test.txt")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="robustness_results")

    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model list, e.g. 1D_Only,2D_Only,CNN_LSTM,Cross_Attention,SE_Dual",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="awgn,rayleigh,rician",
        help="Comma-separated channel list: clean,awgn,rayleigh,rician",
    )
    parser.add_argument(
        "--snrs",
        type=str,
        default="-10,-5,0,5,10",
        help="Comma-separated SNR points in dB.",
    )

    parser.add_argument("--weight_root", type=str, default="runs", help="Root directory to recursively search weights.")
    parser.add_argument("--weights_json", type=str, default=None, help="Optional JSON mapping model_name -> weight_path.")
    parser.add_argument("--k_factor", type=float, default=5.0, help="Rician K-factor.")
    parser.add_argument("--fading_mode", type=str, default="flat", choices=["flat", "fast"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")
    print(f"Save dir: {args.save_dir}")

    model_list = parse_csv_list(args.models, str)
    channel_list = parse_csv_list(args.channels, str)
    snr_list = parse_csv_list(args.snrs, float)

    # Display integer SNRs cleanly if possible.
    snr_list_display = [int(x) if float(x).is_integer() else x for x in snr_list]

    weights_mapping = load_weight_mapping(args.weights_json)

    # Save run config.
    run_config = vars(args).copy()
    run_config["model_list"] = model_list
    run_config["channel_list"] = channel_list
    run_config["snr_list"] = snr_list
    with open(os.path.join(args.save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    # Load dataset once; channel/SNR are changed dynamically.
    dataset = RobustDroneDataset(
        txt_path=args.test_txt,
        N=512,
        T=2096,
        channel_type="awgn",
        snr_db=None,
        k_factor=args.k_factor,
        fading_mode=args.fading_mode,
        base_seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load all available models.
    loaded_models = {}
    used_weights = {}
    missing_weights = {}

    print("\nLoading models...")
    for model_name in model_list:
        if model_name not in MODEL_REGISTRY:
            print(f"[Skip] Unsupported model name: {model_name}")
            continue

        weight_path = find_latest_weight(
            model_name=model_name,
            weight_root=args.weight_root,
            weight_mapping=weights_mapping,
        )

        if weight_path is None:
            print(f"[Warning] No weight found for {model_name}. This model will be skipped.")
            missing_weights[model_name] = None
            continue

        print(f"[Load] {model_name}: {weight_path}")
        model = build_model(model_name, args.num_classes).to(device)
        model = load_weights(model, weight_path, device)
        model.eval()

        loaded_models[model_name] = model
        used_weights[model_name] = weight_path

    with open(os.path.join(args.save_dir, "used_weights.json"), "w", encoding="utf-8") as f:
        json.dump(used_weights, f, indent=2, ensure_ascii=False)

    if missing_weights:
        with open(os.path.join(args.save_dir, "missing_weights.json"), "w", encoding="utf-8") as f:
            json.dump(missing_weights, f, indent=2, ensure_ascii=False)

    if not loaded_models:
        raise RuntimeError("No models were loaded. Please check --weight_root or --weights_json.")

    # Evaluation: channel -> model -> [acc at each snr]
    results = {}

    print("\nStarting robustness evaluation...")
    for channel in channel_list:
        dataset.set_channel(channel)
        results[channel] = {}

        for model_name, model in loaded_models.items():
            results[channel][model_name] = []

        for snr in snr_list:
            dataset.set_snr(snr)

            for model_name, model in loaded_models.items():
                desc = f"{channel.upper()} | {PLOT_LABELS.get(model_name, model_name)} | SNR={snr:g}dB"
                acc = evaluate_accuracy(model, dataloader, device, desc=desc)
                results[channel][model_name].append(acc)

                print(f"{channel:<8} | {model_name:<16} | SNR={snr:>5g} dB | Acc={acc:.2f}%")

    # Save JSON.
    json_results = {
        "snr_db": snr_list,
        "accuracy_by_channel": {
            channel: {
                model_name: [round(float(acc), 4) for acc in acc_list]
                for model_name, acc_list in channel_data.items()
            }
            for channel, channel_data in results.items()
        },
    }
    json_path = os.path.join(args.save_dir, "robustness_all_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    # Save CSV and LaTeX.
    save_csv(results, snr_list, os.path.join(args.save_dir, "robustness_all_results.csv"))
    save_latex_tables(results, snr_list_display, os.path.join(args.save_dir, "robustness_latex_tables.txt"))

    # Save curves.
    plot_channel_curves(results, snr_list, args.save_dir)

    print("\nDone.")
    print(f"Results saved to: {args.save_dir}")
    print(f"Main JSON: {json_path}")


if __name__ == "__main__":
    main()
