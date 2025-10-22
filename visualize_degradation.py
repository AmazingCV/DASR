"""
可视化不同退化参数的效果
用法: python visualize_degradation.py --img_path 你的HR红外图像.png
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import argparse

def apply_degradation(img, sigma, noise_level, scale=2):
    """应用退化：模糊 + 下采样 + 噪声"""
    # 高斯模糊
    if len(img.shape) == 3:
        blurred = np.zeros_like(img, dtype=float)
        for c in range(img.shape[2]):
            blurred[:,:,c] = gaussian_filter(img[:,:,c], sigma=sigma)
    else:
        blurred = gaussian_filter(img, sigma=sigma)
    
    # 下采样
    h, w = img.shape[:2]
    downsampled = cv2.resize(blurred, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
    
    # 添加噪声
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, downsampled.shape)
        downsampled = np.clip(downsampled + noise, 0, 255)
    
    return downsampled.astype(np.uint8)

def visualize_degradations(img_path, scale=2):
    """可视化不同退化参数组合"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    
    # 定义要测试的参数组合
    configs = [
        {"sigma": 0.5, "noise": 0, "label": "轻微模糊\n(sigma=0.5, noise=0)"},
        {"sigma": 1.5, "noise": 0, "label": "中等模糊\n(sigma=1.5, noise=0)"},
        {"sigma": 3.0, "noise": 0, "label": "强模糊\n(sigma=3.0, noise=0)"},
        {"sigma": 1.5, "noise": 10, "label": "中等模糊+轻微噪声\n(sigma=1.5, noise=10)"},
        {"sigma": 2.5, "noise": 15, "label": "强模糊+中等噪声\n(sigma=2.5, noise=15)"},
        {"sigma": 4.0, "noise": 25, "label": "很强模糊+高噪声\n(sigma=4.0, noise=25)"},
    ]
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('不同退化参数效果对比 - 请找出最接近你真实LR图像的配置', fontsize=14, fontweight='bold')
    
    # 原图
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始HR图像', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 各种退化
    for idx, config in enumerate(configs):
        row = (idx + 1) // 4
        col = (idx + 1) % 4
        
        degraded = apply_degradation(img, config['sigma'], config['noise'], scale)
        axes[row, col].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(config['label'], fontsize=9)
        axes[row, col].axis('off')
    
    # 隐藏空白子图
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    output_path = 'degradation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {output_path}")
    print("\n请查看图像，找出最接近你真实LR图像的配置")
    print("然后根据该配置设置训练参数的范围")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='HR红外图像路径')
    parser.add_argument('--scale', type=int, default=2, help='下采样倍数')
    args = parser.parse_args()
    
    visualize_degradations(args.img_path, args.scale)
