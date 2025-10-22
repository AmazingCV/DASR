"""
检查HR-LR配对是否正确
"""
import cv2
import numpy as np
import glob
import os
import argparse

def check_image_pairs(hr_dir, lr_dir, num_check=5):
    """检查HR和LR图像配对"""
    print("="*60)
    print("检查HR-LR图像配对")
    print("="*60)
    
    # 获取文件列表
    hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.*')))
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.*')))
    
    print(f"\n📁 文件统计:")
    print(f"  HR图像数量: {len(hr_files)}")
    print(f"  LR图像数量: {len(lr_files)}")
    
    if len(hr_files) == 0:
        print("\n❌ 错误: HR文件夹为空!")
        return
    if len(lr_files) == 0:
        print("\n❌ 错误: LR文件夹为空!")
        return
    
    # 检查前几对
    num_check = min(num_check, len(hr_files), len(lr_files))
    print(f"\n🔍 检查前{num_check}对图像:\n")
    
    for i in range(num_check):
        hr_path = hr_files[i]
        lr_path = lr_files[i]
        
        hr_img = cv2.imread(hr_path)
        lr_img = cv2.imread(lr_path)
        
        if hr_img is None:
            print(f"  [{i+1}] ❌ 无法读取HR: {os.path.basename(hr_path)}")
            continue
        if lr_img is None:
            print(f"  [{i+1}] ❌ 无法读取LR: {os.path.basename(lr_path)}")
            continue
        
        hr_h, hr_w = hr_img.shape[:2]
        lr_h, lr_w = lr_img.shape[:2]
        scale_h = hr_h / lr_h
        scale_w = hr_w / lr_w
        
        # 计算简单的模糊估计
        hr_lap = cv2.Laplacian(cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        lr_lap = cv2.Laplacian(cv2.cvtColor(lr_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        
        print(f"  [{i+1}] 配对:")
        print(f"      HR: {os.path.basename(hr_path):30s} | 尺寸: {hr_w}x{hr_h} | 清晰度: {hr_lap:8.1f}")
        print(f"      LR: {os.path.basename(lr_path):30s} | 尺寸: {lr_w}x{lr_h} | 清晰度: {lr_lap:8.1f}")
        print(f"      缩放比例: {scale_w:.2f}x (宽) × {scale_h:.2f}x (高)")
        
        # 警告检查
        if abs(scale_h - scale_w) > 0.1:
            print(f"      ⚠️  警告: 宽高缩放比例不一致!")
        if scale_h < 1.5 or scale_w < 1.5:
            print(f"      ⚠️  警告: LR尺寸接近或大于HR，这不正常!")
        if lr_lap > hr_lap:
            print(f"      ⚠️  警告: LR比HR更清晰，配对可能有误!")
        print()
    
    # 统计分析
    print("="*60)
    print("📊 整体统计:")
    scales = []
    blur_ratios = []
    
    for hr_path, lr_path in zip(hr_files[:20], lr_files[:20]):
        hr_img = cv2.imread(hr_path)
        lr_img = cv2.imread(lr_path)
        if hr_img is not None and lr_img is not None:
            scale = hr_img.shape[0] / lr_img.shape[0]
            scales.append(scale)
            
            hr_lap = cv2.Laplacian(cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            lr_lap = cv2.Laplacian(cv2.cvtColor(lr_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if lr_lap > 0:
                blur_ratios.append(hr_lap / lr_lap)
    
    if scales:
        print(f"  平均缩放比例: {np.mean(scales):.2f}x (标准差: {np.std(scales):.2f})")
        print(f"  建议的--scale参数: {int(round(np.mean(scales)))}")
    
    if blur_ratios:
        print(f"  HR/LR清晰度比值: {np.mean(blur_ratios):.2f}x (标准差: {np.std(blur_ratios):.2f})")
        if np.mean(blur_ratios) < 1.5:
            print(f"  ⚠️  清晰度差异较小，LR可能不够模糊")
        elif np.mean(blur_ratios) > 10:
            print(f"  ✅ LR明显比HR模糊，这是正常的")
    
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_path', type=str, required=True, help='HR图像目录')
    parser.add_argument('--lr_path', type=str, required=True, help='LR图像目录')
    parser.add_argument('--num_check', type=int, default=5, help='检查图像对数')
    args = parser.parse_args()
    
    check_image_pairs(args.hr_path, args.lr_path, args.num_check)
