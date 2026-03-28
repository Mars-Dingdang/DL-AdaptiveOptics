# 基于生成模型的计算自适应光学
# English Title: Computational Adaptive Optics with Generative Models

本项目面向课程题目：

- 任务：卫星遥感图像去湍流与增强（Image-to-Image Restoration）
- 输入：受大气湍流、光学模糊、传感器噪声影响的退化图
- 输出：恢复后的清晰图像

项目提供了从数据合成、模型训练、离线评估到在线 Demo 的完整流程，支持两条主线模型：

- Baseline：U-Net
- 主模型：Pix2Pix 风格条件 GAN（带物理一致性约束）

并提供了轻量条件扩散模型骨架，便于后续加分实验。

**GitHub Repo**: .

**Please follow this workflow since main branch is protected**:
```bash
git clone <repo_url>
cd Project

git checkout main
git pull origin main
git checkout -b your_branch_name # each time you start a new feature/experiment

git add .
git commit -m "Your commit message"
git push origin your_branch_name
# After PR is merged, switch back to main and pull latest
git checkout main
git pull origin main
```

---

## 1. 技术路线概览

### 1.1 退化建模（Physics-inspired Degradation）

在训练时不依赖稀缺的真实配对数据，而是对清晰遥感图进行在线退化，生成配对样本：

- Zernike 多项式生成相位屏（wavefront phase screen）
- 通过傅里叶光学将相位屏转换为 PSF（点扩散函数）
- PSF 卷积模拟光学模糊
- 可选运动模糊
- Poisson + Gaussian 噪声模拟传感器扰动
- JPEG 压缩伪影模拟成像链路失真

### 1.2 模型架构

1) U-Net Baseline

- 编码器-解码器结构
- 跳跃连接保留细节
- 适合作为稳定保底模型

2) 条件 GAN（Pix2Pix 风格）

- Generator：U-Net 风格生成器
- Discriminator：PatchGAN 判别器
- 目标函数包含对抗项 + 重建项 + 物理一致性项

3) 条件扩散骨架

- Sinusoidal 时间嵌入
- 条件去噪网络
- 噪声预测目标（MSE）
- DDIM 采样接口

### 1.3 训练方法

1) U-Net 训练目标

- 监督学习，主损失为 L1 重建损失

2) GAN 训练目标

- 判别器损失：BCE(real/fake)
- 生成器损失：
  - 对抗损失
  - L1 重建损失
  - 退化一致性损失（将生成图做前向退化后，应逼近输入退化图）

3) 评估指标

- PSNR
- SSIM
- LPIPS（可选）

---

## 2. 项目文件树

```text
Project/
├─ README.md
├─ initial_plan.md
├─ initial_prompt.md
├─ requirements.txt
├─ train.py
├─ eval.py
├─ configs/
│  └─ default.yaml
├─ data/
│  ├─ __init__.py
│  ├─ get_data.py
│  └─ dataset.py
├─ modules/
│  ├─ __init__.py
│  ├─ baseline_unet.py
│  ├─ gan_models.py
│  └─ diffusion.py
├─ utils/
│  ├─ __init__.py
│  ├─ degradation.py
│  ├─ metrics.py
│  └─ visualization.py
└─ demo/
   └─ app.py
```

---

## 3. 每个文件的作用、实现方式与原理

### 3.1 根目录

- README.md
  - 作用：项目说明、运行指南
  - 实现方式：文档化项目目标、结构和命令
  - 原理：保证团队可复现和可交接

- initial_plan.md
  - 作用：课程项目策划与阶段建议
  - 实现方式：记录阶段目标、风险和展示策略
  - 原理：项目管理与评分导向对齐

- initial_prompt.md
  - 作用：开发约束与目录规范
  - 实现方式：定义 Phase 要求和代码规范
  - 原理：将需求转化为工程实现清单

- requirements.txt
  - 作用：统一依赖管理
  - 实现方式：列出 PyTorch、OpenCV、Gradio 等依赖
  - 原理：环境可复现

- train.py
  - 作用：统一训练入口
  - 实现方式：
    - 读取 YAML 配置
    - 构建数据集与 DataLoader
    - 支持 unet/gan 两种训练循环
    - 指标监控与 checkpoint 保存
  - 原理：配置驱动 + 模型解耦

- eval.py
  - 作用：统一离线评估入口
  - 实现方式：
    - 加载 checkpoint（UNet 或 GAN）
    - 在验证集计算 PSNR/SSIM/LPIPS
    - 可选保存可视化样例
  - 原理：训练与评估分离，便于报告复现

### 3.2 configs

- configs/default.yaml
  - 作用：集中管理超参数和路径
  - 实现方式：包含数据路径、退化参数、模型类型、优化器、调度器等
  - 原理：避免硬编码，便于实验对比

### 3.3 data

- data/get_data.py
  - 作用：下载和准备公开遥感数据
  - 实现方式：
    - 支持 UC Merced 自动下载与解压
    - 预留 WHU-RS19 / NWPU-VHR10 手动下载提示
  - 原理：标准化数据入口，减少人工操作错误

- data/dataset.py
  - 作用：构建训练样本对 (degraded, clear)
  - 实现方式：
    - 扫描图片
    - resize/crop/flip 预处理
    - 在 __getitem__ 中调用退化模拟实时生成退化图
  - 原理：在线退化增强数据多样性，避免提前离线生成占用大量存储

### 3.4 modules

- modules/baseline_unet.py
  - 作用：U-Net 基线模型
  - 实现方式：DoubleConv + Down + Up + skip connection
  - 原理：多尺度特征融合，提升恢复细节能力

- modules/gan_models.py
  - 作用：条件 GAN 主模型与损失定义
  - 实现方式：
    - Generator：Pix2Pix 风格 U-Net
    - Discriminator：PatchGAN
    - 物理一致性损失：对生成图进行近似退化并约束与输入一致
  - 原理：
    - 对抗学习提升感知质量
    - L1 保证内容保真
    - 物理约束抑制不合理“幻觉细节”

- modules/diffusion.py
  - 作用：条件扩散模型骨架
  - 实现方式：
    - 前向扩散 q_sample
    - 噪声预测目标 p_losses
    - DDIM 采样 sample_ddim
  - 原理：通过逐步去噪建模复杂条件分布

### 3.5 utils

- utils/degradation.py
  - 作用：大气湍流与成像退化核心模块
  - 实现方式：Zernike 相位屏 + PSF + 噪声 + 压缩伪影
  - 原理：将 AO 场景中的波前畸变与传感器噪声过程转化为可训练的前向退化模型

- utils/metrics.py
  - 作用：评估指标计算
  - 实现方式：
    - skimage 计算 PSNR/SSIM
    - 可选 LPIPS（懒加载）
  - 原理：同时评估失真与感知质量

- utils/visualization.py
  - 作用：结果可视化与导出
  - 实现方式：保存 Input/GT/Prediction 三联图
  - 原理：为报告与海报提供直观证据

### 3.6 demo

- demo/app.py
  - 作用：交互式展示应用
  - 实现方式：
    - Gradio 页面上传图像
    - 加载 checkpoint 推理
    - Image Slider 前后对比（缺插件时自动回退双图）
  - 原理：提高现场展示效果与可解释性

---

## 4. 从零开始到成功运行：完整命令行流程

下面按 Windows 环境给出完整命令。推荐在项目根目录执行。

### 4.1 进入项目目录

```powershell
cd C:/Users/23826/Desktop/university/Grade1-2/DL/Project
```

### 4.2 创建并激活 Conda 环境

```powershell
conda create -n DLProject python=3.12 -y
conda activate DLProject
```

如果你的 shell 中 conda activate 不生效，可以先执行：

```bash
source D:/Anaconda/etc/profile.d/conda.sh
conda activate DLProject
```

### 4.3 安装 PyTorch（GPU）

请根据你机器 CUDA 版本选择官方命令。示例（CUDA 12.4）：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

如果你只用 CPU：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4.4 安装项目依赖

```powershell
python -m pip install -r requirements.txt
```

### 4.5 验证环境

```powershell
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import cv2, yaml, skimage, gradio; print('deps_ok')"
```

### 4.6 下载数据（UC Merced）

```powershell
python data/get_data.py --dataset uc_merced
```

下载完成后，默认训练路径为：

- data/raw/UCMerced_LandUse/Images

### 4.7 开始训练（U-Net 默认配置）

```powershell
python train.py --config configs/default.yaml
```

训练输出：

- checkpoints/best_unet.pt
- checkpoints/unet_epoch_*.pt

### 4.8 训练 GAN（可选）

先复制一份配置并把模型类型改成 gan：

```powershell
Copy-Item configs/default.yaml configs/train_gan.yaml
(Get-Content configs/train_gan.yaml) -replace 'type: unet', 'type: gan' | Set-Content configs/train_gan.yaml
python train.py --config configs/train_gan.yaml
```

训练输出：

- checkpoints/best_gan.pt
- checkpoints/gan_epoch_*.pt

### 4.9 离线评估

评估 U-Net：

```powershell
python eval.py --config configs/default.yaml --checkpoint checkpoints/best_unet.pt --save-images --out-dir outputs/eval_unet
```

评估 GAN：

```powershell
python eval.py --config configs/default.yaml --checkpoint checkpoints/best_gan.pt --model-type gan --save-images --out-dir outputs/eval_gan
```

评估结果：

- outputs/eval_*/metrics.txt
- outputs/eval_*/samples/*.png

### 4.10 启动 Demo

启动 U-Net Demo：

```powershell
python demo/app.py --config configs/default.yaml --checkpoint checkpoints/best_unet.pt --model-type unet --host 127.0.0.1 --port 7860
```

启动 GAN Demo：

```powershell
python demo/app.py --config configs/default.yaml --checkpoint checkpoints/best_gan.pt --model-type gan --host 127.0.0.1 --port 7860
```

浏览器打开：

- http://127.0.0.1:7860

---

## 5. 训练配置建议（RTX 5090 单卡）

可在 configs/default.yaml 中调整：

- data.batch_size：建议 16 或 24（视显存）
- train.epochs：先 10 做 smoke test，再拉到 50+
- model.base_channels：显存充足可增大
- data.num_workers：建议 8 或 12

建议流程：

1. 先用 U-Net 快速跑通全链路（训练-评估-Demo）
2. 再切 GAN 做展示主模型
3. 最后将 diffusion.py 作为加分实验骨架扩展

---

## 6. 常见问题

1) 报错：无法导入 gradio

```powershell
python -m pip install gradio gradio-imageslider
```

2) 报错：找不到数据目录

- 检查 configs/default.yaml 的 data.train_root 是否与实际路径一致

3) CUDA 不可用

- 检查显卡驱动/CUDA
- 重新安装匹配版本的 PyTorch

4) 训练速度慢

- 提升 batch_size 与 num_workers
- 降低 image_size 做快速验证

---

## 7. 一键最小可运行命令（最短路径）

```powershell
cd C:/Users/23826/Desktop/university/Grade1-2/DL/Project
conda activate DLProject
python -m pip install -r requirements.txt
python data/get_data.py --dataset uc_merced
python train.py --config configs/default.yaml
python demo/app.py --config configs/default.yaml --checkpoint checkpoints/best_unet.pt
```

完成以上流程后，即可实现从数据到训练再到展示的闭环运行。

---

## 8. Linux Bash 命令版本（云端训练）

下面给出与上文 Windows PowerShell 对应的 Linux Bash 版本命令。

### 8.1 进入项目目录

```bash
cd /path/to/Project
```

### 8.2 创建并激活 Conda 环境

```bash
conda create -n DLProject python=3.12 -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate DLProject
```

如果你的服务器是 Miniconda，可改为：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DLProject
```

### 8.3 安装 PyTorch（GPU）

按你的 CUDA 版本安装，示例（CUDA 12.4）：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

如果只用 CPU：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 8.4 安装项目依赖

```bash
python -m pip install -r requirements.txt
```

### 8.5 验证环境

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import cv2, yaml, skimage, gradio; print('deps_ok')"
nvidia-smi
```

### 8.6 下载数据（UC Merced）

```bash
python data/get_data.py --dataset uc_merced
```

### 8.7 开始训练（U-Net 默认配置）

```bash
python train.py --config configs/default.yaml
```

### 8.8 训练 GAN（可选）

```bash
cp configs/default.yaml configs/train_gan.yaml
sed -i 's/type: unet/type: gan/' configs/train_gan.yaml
python train.py --config configs/train_gan.yaml
```

如果你的系统是 macOS（BSD sed），请用：

```bash
sed -i '' 's/type: unet/type: gan/' configs/train_gan.yaml
```

### 8.9 离线评估

评估 U-Net：

```bash
python eval.py --config configs/default.yaml --checkpoint checkpoints/best_unet.pt --save-images --out-dir outputs/eval_unet
```

评估 GAN：

```bash
python eval.py --config configs/default.yaml --checkpoint checkpoints/best_gan.pt --model-type gan --save-images --out-dir outputs/eval_gan
```

### 8.10 启动 Demo

```bash
python demo/app.py --config configs/default.yaml --checkpoint checkpoints/best_unet.pt --model-type unet --host 0.0.0.0 --port 7860
```

如果需要公网访问：

```bash
python demo/app.py --config configs/default.yaml --checkpoint checkpoints/best_unet.pt --model-type unet --host 0.0.0.0 --port 7860 --share
```

### 8.11 Linux 一键最小可运行命令（最短路径）

```bash
cd /path/to/Project
source ~/anaconda3/etc/profile.d/conda.sh
conda activate DLProject
python -m pip install -r requirements.txt
python data/get_data.py --dataset uc_merced
python train.py --config configs/default.yaml
python demo/app.py --config configs/default.yaml --checkpoint checkpoints/best_unet.pt --host 0.0.0.0 --port 7860
```
