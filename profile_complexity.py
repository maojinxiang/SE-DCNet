import os
import time
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

from dataset import DroneDataset
from model import (
    DualChannelDroneNet,
    DroneNet_1D_Only,
    DroneNet_2D_Only,
    DroneNet_ResNet_Only,
    DroneNet_CNN_LSTM,
    DroneNet_CrossAttention,
    DualChannelConcatNet,
    DualChannelWeightedFusionNet,
)


# ===========================
# 直接改这里
# ===========================
NUM_CLASSES = 9
TEST_TXT = "test.txt"
BATCH_SIZE = 1
NUM_WORKERS = 0
WARMUP = 10
REPEAT = 50
SAVE_CSV = "complexity_results.csv"

MODELS = {
    "IQ-CNN": DroneNet_1D_Only,
    "STFT-ResNet": DroneNet_ResNet_Only,
    "STFT-EfficientNet": DroneNet_2D_Only,
    "CNN-LSTM": DroneNet_CNN_LSTM,
    "Fusion-Concat": DualChannelConcatNet,
    "Fusion-Weighted": DualChannelWeightedFusionNet,
    "Cross-Attention": DroneNet_CrossAttention,
    "SE-DCNet": DualChannelDroneNet,
}

# 如果你要把 accuracy 也放进表格，手动填这里
ACCURACY = {
    "IQ-CNN": 94.98,
    "STFT-ResNet": 93.96,
    "STFT-EfficientNet": 96.48,
    "CNN-LSTM": 97.53,
    "Fusion-Concat": 97.29,       # 改成你的真实结果
    "Fusion-Weighted": 96.23,     # 改成你的真实结果
    "Cross-Attention": 97.82,
    "SE-DCNet": 98.94,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_one_batch(device):
    dataset = DroneDataset(TEST_TXT, N=512, T=2096, train_mode=False)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    inputs, labels = next(iter(loader))
    img_input, seq_input = inputs

    img_input = img_input.to(device)
    seq_input = seq_input.to(device)

    return img_input, seq_input


@torch.no_grad()
def measure_inference_time(model, img_input, seq_input, device):
    model.eval()

    # warmup
    for _ in range(WARMUP):
        _ = model(img_input, seq_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(REPEAT):
        _ = model(img_input, seq_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    avg_time_ms = (end - start) * 1000.0 / REPEAT / BATCH_SIZE
    return avg_time_ms


def compute_macs(model, img_input, seq_input):
    if not HAS_THOP:
        return None

    model.eval()

    try:
        macs, params = profile(
            model,
            inputs=(img_input, seq_input),
            verbose=False,
        )
        # thop 返回的是 MACs，不是严格 FLOPs
        return macs / 1e9
    except Exception as e:
        print(f"[WARN] THOP failed: {repr(e)}")
        return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not os.path.exists(TEST_TXT):
        raise FileNotFoundError(f"{TEST_TXT} not found.")

    img_input, seq_input = get_one_batch(device)

    print(f"[INFO] img_input shape: {tuple(img_input.shape)}")
    print(f"[INFO] seq_input shape: {tuple(seq_input.shape)}")

    rows = []

    for name, model_cls in MODELS.items():
        print("\n" + "=" * 60)
        print(f"[INFO] Profiling: {name}")
        print("=" * 60)

        model = model_cls(num_classes=NUM_CLASSES).to(device)
        model.eval()

        params_m = count_params(model) / 1e6
        print(f"Params: {params_m:.3f} M")

        macs_g = compute_macs(model, img_input, seq_input)
        if macs_g is not None:
            print(f"MACs: {macs_g:.3f} G")
        else:
            print("MACs: N/A")

        infer_ms = measure_inference_time(model, img_input, seq_input, device)
        print(f"Inference time: {infer_ms:.3f} ms/sample")

        rows.append({
            "Method": name,
            "Params (M)": round(params_m, 3),
            "MACs (G)": round(macs_g, 3) if macs_g is not None else "N/A",
            "Inference Time (ms/sample)": round(infer_ms, 3),
            "Accuracy (%)": ACCURACY.get(name, "N/A"),
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("Complexity Results")
    print("=" * 60)
    print(df.to_string(index=False))

    df.to_csv(SAVE_CSV, index=False)
    print(f"\n[DONE] Saved to: {SAVE_CSV}")

    # 输出 LaTeX 表格，方便复制到论文
    print("\nLaTeX table:")
    print(df.to_latex(index=False, escape=False))


if __name__ == "__main__":
    main()