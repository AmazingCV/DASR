"""
分析红外图像的退化参数
用法: python analyze_degradation.py --hr_path HR图像路径 --lr_path LR图像路径
"""
import cv2
import numpy as np
from scipy import signal
from scipy.optimize import minimize
import argparse

def estimate_blur_kernel(hr_img, lr_img, scale=2):
    """估计模糊核和噪声水平"""
    # 将HR下采样到LR尺寸
    h, w = lr_img.shape[:2]
    hr_downsampled = cv2.resize(hr_img, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 转换为灰度
    if len(hr_downsampled.shape) == 3:
        hr_gray = cv2.cvtColor(hr_downsampled, cv2.COLOR_BGR2GRAY)
        lr_gray = cv2.cvtColor(lr_img, cv2.COLOR_BGR2GRAY)
    else:
        hr_gray = hr_downsampled
        lr_gray = lr_img
    
    # 计算频域差异来估计模糊
    hr_fft = np.fft.fft2(hr_gray)
    lr_fft = np.fft.fft2(lr_gray)
    
    # 估计传递函数
    H = lr_fft / (hr_fft + 1e-10)
    
    # 分析频谱衰减来估计sigma
    freq_decay = np.abs(H)
    radius = np.min(freq_decay.shape) // 4
    center = (freq_decay.shape[0]//2, freq_decay.shape[1]//2)
    
    # 径向平均
    y, x = np.ogrid[:freq_decay.shape[0], :freq_decay.shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    radial_mean = np.zeros(radius)
    for i in range(radius):
        mask = (r == i)
        radial_mean[i] = freq_decay[mask].mean()
    
    # 拟合高斯衰减估计sigma
    def gaussian_decay(sigma, freq):
        return np.exp(-2 * (np.pi * sigma * freq)**2)
    
    freq_range = np.arange(radius) / freq_decay.shape[0]
    
    def objective(sigma):
        predicted = gaussian_decay(sigma, freq_range)
        return np.sum((radial_mean - predicted)**2)
    
    result = minimize(objective, x0=2.0, bounds=[(0.1, 20.0)])  # 扩大上限到20
    estimated_sigma = result.x[0]
    
    # 估计噪声水平
    diff = hr_downsampled.astype(float) - lr_img.astype(float)
    noise_std = np.std(diff)
    
    return estimated_sigma, noise_std

def analyze_multiple_pairs(hr_dir, lr_dir, num_samples=10):
    """分析多对图像，给出统计结果"""
    import os
    import glob
    
    hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.*')))[:num_samples]
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.*')))[:num_samples]
    
    sigmas = []
    noises = []
    
    for hr_path, lr_path in zip(hr_files, lr_files):
        print(f"分析: {os.path.basename(hr_path)}")
        hr_img = cv2.imread(hr_path)
        lr_img = cv2.imread(lr_path)
        
        if hr_img is None or lr_img is None:
            continue
            
        sigma, noise = estimate_blur_kernel(hr_img, lr_img)
        sigmas.append(sigma)
        noises.append(noise)
        print(f"  估计sigma: {sigma:.2f}, 噪声std: {noise:.2f}")
    
    if sigmas:
        print("\n" + "="*50)
        print("统计结果:")
        print(f"Sigma范围: [{np.min(sigmas):.2f}, {np.max(sigmas):.2f}]")
        print(f"Sigma均值: {np.mean(sigmas):.2f} ± {np.std(sigmas):.2f}")
        print(f"噪声范围: [{np.min(noises):.2f}, {np.max(noises):.2f}]")
        print(f"噪声均值: {np.mean(noises):.2f} ± {np.std(noises):.2f}")
        print("="*50)
        print("\n🎯 建议的完整训练参数:")
        print("-" * 50)
        
        # 基础参数
        print("# 基础参数")
        print(f"--scale='2'  # 根据你的HR和LR尺寸比例调整")
        print(f"--blur_kernel=21  # 通常不需要修改")
        print()
        
        # 模糊类型
        print("# 模糊类型（根据视觉判断）")
        print(f"--blur_type='iso_gaussian'  # 如无方向性模糊用这个")
        print(f"# --blur_type='aniso_gaussian'  # 如有运动模糊用这个")
        print()
        
        # 模糊强度
        print("# 模糊强度（根据分析结果）")
        sig_min_suggest = max(0.2, np.min(sigmas) - 0.5)
        sig_max_suggest = min(8.0, np.max(sigmas) + 0.5)
        print(f"--sig_min={sig_min_suggest:.1f}")
        print(f"--sig_max={sig_max_suggest:.1f}")
        print()
        
        # 噪声
        print("# 噪声水平（根据分析结果）")
        noise_suggest = np.mean(noises)
        print(f"--noise={noise_suggest:.1f}")
        print()
        
        # 下采样方式
        print("# 下采样方式")
        print(f"--mode='bicubic'  # 通常用这个")
        print("-" * 50)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_path', type=str, required=True, help='HR图像目录')
    parser.add_argument('--lr_path', type=str, required=True, help='LR图像目录')
    parser.add_argument('--num_samples', type=int, default=10, help='分析样本数量')
    args = parser.parse_args()
    
    analyze_multiple_pairs(args.hr_path, args.lr_path, args.num_samples)
