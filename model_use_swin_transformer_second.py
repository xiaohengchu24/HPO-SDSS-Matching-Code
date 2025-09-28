import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import cv2
from torchvision import transforms
import timm
import shutil
from concurrent.futures import ThreadPoolExecutor
import warnings
from tqdm import tqdm
import time  # 新增导入
```python
# ==============================================================================
# Code Source Acknowledgment
# ==============================================================================
# This script contains modifications based on the original Swin Transformer
# architecture, specifically the 'swin_tiny_patch4_window7_224' module,
# as proposed in the publication:
# Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer using
# Shifted Windows. 18th IEEE/CVF International Conference on Computer Vision (ICCV).
#
# The original code's structure was adapted and modified by Xiao Hengchu
# for integration with the SDSS astronomical data pipeline and streak detection optimization.
# ==============================================================================
# 忽略警告
warnings.filterwarnings('ignore')

# 配置参数
MODEL_PATH = r"O:/pytorch_model.bin"  # 训练好的模型权重
INPUT_DIR = r"G:/sdss_dr17/"  # 输k入目录
OUTPUT_DIR = r"G:/sdss_dr18/"  # 输出目录
OUTPUT_LIST = r"O:/streak_files_2.txt"  # 阳性文件列表
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
CONFIDENCE_THRESHOLD = 0.5
POSITIVE_PATCH_THRESHOLD = 3
BATCH_SIZE = 16
NUM_WORKERS = 4

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0),
    transforms.RandomVerticalFlip(p=0),
    transforms.RandomRotation(0),
    transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(1, 1)),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

# 加载模型
def load_model():
    try:
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2, img_size=256)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"已加载模型权重：{MODEL_PATH}")
        else:
            raise FileNotFoundError(f"未找到模型权重文件：{MODEL_PATH}")
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"加载模型时出错：{str(e)}")
        raise

# 切割 FITS 图像为 256x256 patch
def split_image_to_patches(image, patch_size=256):
    height, width = image.shape
    patches = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = np.zeros((patch_size, patch_size), dtype=np.float32)
            h_end = min(y + patch_size, height)
            w_end = min(x + patch_size, width)
            patch[:h_end - y, :w_end - x] = image[y:h_end, x:w_end]
            patches.append(patch)
    return patches

# 预处理 patch
def preprocess_patch(patch):
    patch = np.clip(patch, np.percentile(patch, 1), np.percentile(patch, 99))
    patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
    patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))
    patch = np.stack([patch] * 3, axis=0)
    return torch.from_numpy(patch).float()

# 批处理预测
def predict_batch(model, patches, transform, device):
    patches = torch.stack([transform(patch) for patch in patches]).to(device, non_blocking=True)
    with torch.no_grad():
        outputs = model(patches)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)[:, 1]
    return predicted.cpu().numpy(), probs.cpu().numpy()

# 处理单张 FITS 文件
def process_fits_file(fpath, model, transform, device, confidence_threshold, positive_patch_threshold):
    try:
        with fits.open(fpath) as hdul:
            for hdu in hdul:
                if hdu.header.get('NAXIS', 0) >= 2 and hdu.data is not None:
                    data = hdu.data
                    if data.ndim > 2:
                        data = data[0]
                    if data.ndim == 2:
                        data = np.flipud(data).astype(np.float32)
                        break
            else:
                raise ValueError("FITS 文件中未找到有效的 2D 图像")
    except Exception as e:
        print(f"加载 {fpath} 时出错：{str(e)}")
        return False, []

    # 切割为 patch
    patches = split_image_to_patches(data, patch_size=IMG_SIZE)
    if not patches:
        print(f"{fpath} 无有效 patch")
        return False, []

    # 批处理预测
    positive_patches = []
    positive_count = 0
    for i in range(0, len(patches), BATCH_SIZE):
        batch_patches = patches[i:i + BATCH_SIZE]
        batch_patches = [preprocess_patch(patch) for patch in batch_patches]
        predicted, probs = predict_batch(model, batch_patches, transform, device)

        for j, (pred, prob) in enumerate(zip(predicted, probs)):
            if pred == 1 and prob >= confidence_threshold:
                positive_count += 1
                positive_patches.append((i + j, prob))
            if positive_count >= positive_patch_threshold:
                return True, positive_patches
    return positive_count >= positive_patch_threshold, positive_patches

# 主函数：分类并挑出 streak 文件（流水线方式）
def classify_and_extract_streaks(input_dir, output_dir, output_list, model, transform, device, confidence_threshold=0.5,
                                 positive_patch_threshold=1):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录：{output_dir}")

    # 清空输出列表文件（首次运行）
    with open(output_list, 'w') as f:
        pass

    # 无限循环，监控输入目录
    while True:
        # 获取所有 FITS 文件
        fits_files = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(".fits.bz2")
        ]
        if not fits_files:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 警告：未在 {input_dir} 中找到有效的 FITS 文件，等待5分钟...")
            time.sleep(300)  # 等待5分钟
            continue

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 找到 {len(fits_files)} 张 FITS 图像，开始分类...")

        # 流水线处理每个文件
        def process_file(fpath):
            print(f"处理图像: {fpath}")
            is_positive, positive_patches = process_fits_file(fpath, model, transform, device, confidence_threshold,
                                                              positive_patch_threshold)
            print(f"  阳性 patch 数量: {len(positive_patches)}")
            if is_positive:
                print(f"  预测结果: streak (阳性)")
                dest_path = os.path.join(output_dir, os.path.basename(fpath))
                try:
                    shutil.move(fpath, dest_path)
                    print(f"  移动 streak 文件至: {dest_path}")
                    with open(output_list, 'a') as f:
                        f.write(f"{dest_path}\n")  # 记录移动后的路径
                except Exception as e:
                    print(f"  移动文件 {fpath} 失败：{str(e)}")
            else:
                print(f"  预测结果: no_streak (阴性)")
                try:
                    os.remove(fpath)
                    print(f"  已删除阴性文件：{fpath}")
                except Exception as e:
                    print(f"  删除文件 {fpath} 失败：{str(e)}")

        # 使用 ThreadPoolExecutor 进行并行处理
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            list(tqdm(executor.map(process_file, fits_files), total=len(fits_files), desc="分类图像"))

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 已完成当前批次处理，等待5分钟后继续检查新文件...")
        time.sleep(300)  # 处理完后等待5分钟

# 执行分类
def main():
    try:
        # 加载模型
        model = load_model()

        # 分类并挑出 streak 文件
        classify_and_extract_streaks(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            output_list=OUTPUT_LIST,
            model=model,
            transform=transform,
            device=DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            positive_patch_threshold=POSITIVE_PATCH_THRESHOLD
        )
    except KeyboardInterrupt:
        print("用户中断程序，退出...")
    except Exception as e:
        print(f"程序运行出错：{str(e)}")

if __name__ == "__main__":
    main()