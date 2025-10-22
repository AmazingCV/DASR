#!/bin/bash
# 快速参数验证脚本 - 只训练少量epoch来测试不同参数

DATA_PATH="你的数据路径"
PRETRAIN_PATH="experiment/blindsr_x2_bicubic_iso/model/model_600.pt"

# 测试配置1: 小模糊 + 无噪声
echo "测试配置1: 小模糊范围"
python main.py --dir_data=$DATA_PATH \
               --model='blindsr' \
               --scale='2' \
               --save='test_small_blur' \
               --pre_train=$PRETRAIN_PATH \
               --sig_min=0.2 --sig_max=2.0 --noise=0.0 \
               --epochs_encoder=5 --epochs_sr=20 \
               --test_every=500 \
               --lr_encoder=5e-4 --lr_sr=5e-5

# 测试配置2: 中等模糊 + 中等噪声
echo "测试配置2: 中等模糊+噪声"
python main.py --dir_data=$DATA_PATH \
               --model='blindsr' \
               --scale='2' \
               --save='test_medium_blur_noise' \
               --pre_train=$PRETRAIN_PATH \
               --sig_min=1.0 --sig_max=4.0 --noise=15.0 \
               --epochs_encoder=5 --epochs_sr=20 \
               --test_every=500 \
               --lr_encoder=5e-4 --lr_sr=5e-5

# 测试配置3: 大模糊 + 高噪声
echo "测试配置3: 大模糊+高噪声"
python main.py --dir_data=$DATA_PATH \
               --model='blindsr' \
               --scale='2' \
               --save='test_large_blur_noise' \
               --pre_train=$PRETRAIN_PATH \
               --sig_min=2.0 --sig_max=6.0 --noise=25.0 \
               --epochs_encoder=5 --epochs_sr=20 \
               --test_every=500 \
               --lr_encoder=5e-4 --lr_sr=5e-5

echo "快速测试完成！请对比三个配置在验证集上的PSNR/SSIM"
echo "选择效果最好的配置进行完整训练"
