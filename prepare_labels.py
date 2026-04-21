import os
import random


def discover_categories(data_root):
    if not os.path.exists(data_root):
        return []
    cats = []
    for name in os.listdir(data_root):
        full = os.path.join(data_root, name)
        if os.path.isdir(full) and name.startswith('T'):
            cats.append(name)
    return sorted(cats)


def generate_augmented_labels(data_root):
    categories = discover_categories(data_root)
    if not categories:
        print(f"Error: no categories discovered in {data_root}")
        return
    all_samples = []

    # --- 核心修改开始 ---
    # 论文设定：每个样本 10ms (1百万点)。
    # 论文规模：每类约 650 个样本。假设每类有 ~25-30 个原始文件。
    # 计算：650 / 25 = 26。因此我们将切片数设为 25 左右。
    num_slices_per_file = 95

    # 增大步长，让这 25 个切片均匀分布在 1亿个点(100M)的文件中
    # 100M / 25 ≈ 4M。为了安全起见（防止越界），设为 3M。
    stride = 1000000
    # --- 核心修改结束 ---

    print(f"正在生成标签... 目标类别: {categories}")

    for label_idx, cat_name in enumerate(categories):
        cat_path = os.path.join(data_root, cat_name)
        if not os.path.exists(cat_path):
            print(f"Warning: {cat_path} not found.")
            continue

        mat_files = [f for f in os.listdir(cat_path) if f.endswith('.mat')]
        print(f"类别 {cat_name}: 找到 {len(mat_files)} 个源文件")

        for mat_file in mat_files:
            file_path = os.path.abspath(os.path.join(cat_path, mat_file))
            for i in range(num_slices_per_file):
                offset = i * stride
                # 记录格式: 路径, 类别索引, 读取偏移量
                all_samples.append(f"{file_path},{label_idx},{offset}")

    # 打乱数据
    random.seed(42)
    random.shuffle(all_samples)

    total = len(all_samples)
    if total == 0:
        print("Error: 没有生成任何样本，请检查路径。")
        return

    # 划分数据集 (6:2:2)
    train_split = int(total * 0.6)
    val_split = int(total * 0.8)

    with open('train.txt', 'w') as f:
        f.write('\n'.join(all_samples[:train_split]))
    with open('val.txt', 'w') as f:
        f.write('\n'.join(all_samples[train_split:val_split]))
    with open('test.txt', 'w') as f:
        f.write('\n'.join(all_samples[val_split:]))

    print(f"生成完成！总样本数: {total} (每类约 {total // len(categories)} 个)")
    print(f"训练集: {train_split}, 验证集: {val_split - train_split}, 测试集: {total - val_split}")


if __name__ == "__main__":
    # 请确保路径正确指向你的数据集根目录
    generate_augmented_labels(r'./Dataset')