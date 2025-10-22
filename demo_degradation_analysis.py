"""
演示退化分析原理的简单示例
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def create_test_case():
    """创建一个已知退化参数的测试案例"""
    # 创建一个简单的测试图像
    size = 256
    img = np.zeros((size, size))
    
    # 添加一些高频内容（棋盘格）
    for i in range(0, size, 16):
        for j in range(0, size, 16):
            if (i//16 + j//16) % 2 == 0:
                img[i:i+16, j:j+16] = 255
    
    # 已知的退化参数
    true_sigma = 2.5
    true_noise = 15.0
    
    print(f"创建测试图像，真实参数: sigma={true_sigma}, noise={true_noise}")
    
    # 应用已知退化
    hr = img.copy()
    
    # 模糊
    blurred = gaussian_filter(hr, sigma=true_sigma)
    
    # 下采样
    lr = cv2.resize(blurred, (size//2, size//2), interpolation=cv2.INTER_CUBIC)
    lr = cv2.resize(lr, (size, size), interpolation=cv2.INTER_CUBIC)  # 上采样回来方便比较
    
    # 添加噪声
    noise = np.random.normal(0, true_noise, lr.shape)
    lr_noisy = np.clip(lr + noise, 0, 255)
    
    return hr, lr_noisy, true_sigma, true_noise

def estimate_parameters(hr, lr):
    """使用频域分析估计参数"""
    # 频域分析
    hr_fft = np.fft.fft2(hr)
    lr_fft = np.fft.fft2(lr)
    
    # 传递函数
    H = lr_fft / (hr_fft + 1e-10)
    freq_decay = np.abs(H)
    
    # 径向平均
    h, w = freq_decay.shape
    center = (h//2, w//2)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    max_radius = min(h, w) // 4
    radial_profile = []
    frequencies = []
    
    for i in range(max_radius):
        mask = (r == i)
        if mask.any():
            radial_profile.append(freq_decay[mask].mean())
            frequencies.append(i / max(h, w))
    
    radial_profile = np.array(radial_profile)
    frequencies = np.array(frequencies)
    
    # 拟合高斯衰减
    from scipy.optimize import minimize
    
    def objective(sigma):
        predicted = np.exp(-2 * (np.pi * sigma * frequencies)**2)
        return np.sum((radial_profile - predicted)**2)
    
    result = minimize(objective, x0=2.0, bounds=[(0.1, 10.0)])
    estimated_sigma = result.x[0]
    
    # 估计噪声
    diff = hr - lr
    estimated_noise = np.std(diff)
    
    return estimated_sigma, estimated_noise, radial_profile, frequencies

# 运行测试
print("="*60)
print("退化参数分析演示")
print("="*60)

hr, lr, true_sigma, true_noise = create_test_case()
est_sigma, est_noise, radial_profile, frequencies = estimate_parameters(hr, lr)

print(f"\n真实参数:")
print(f"  Sigma: {true_sigma:.2f}")
print(f"  Noise: {true_noise:.2f}")

print(f"\n估计参数:")
print(f"  Sigma: {est_sigma:.2f} (误差: {abs(est_sigma-true_sigma)/true_sigma*100:.1f}%)")
print(f"  Noise: {est_noise:.2f} (误差: {abs(est_noise-true_noise)/true_noise*100:.1f}%)")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(hr, cmap='gray')
axes[0].set_title('原始HR图像')
axes[0].axis('off')

axes[1].imshow(lr, cmap='gray')
axes[1].set_title(f'退化LR图像\n(σ={true_sigma}, noise={true_noise})')
axes[1].axis('off')

# 频谱衰减曲线
predicted = np.exp(-2 * (np.pi * est_sigma * frequencies)**2)
axes[2].plot(frequencies, radial_profile, 'b-', label='实际衰减', linewidth=2)
axes[2].plot(frequencies, predicted, 'r--', label=f'拟合曲线 (σ={est_sigma:.2f})', linewidth=2)
axes[2].set_xlabel('归一化频率')
axes[2].set_ylabel('频谱衰减')
axes[2].set_title('频域分析')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('degradation_analysis_demo.png', dpi=150)
print(f"\n可视化结果已保存到: degradation_analysis_demo.png")
print("="*60)
