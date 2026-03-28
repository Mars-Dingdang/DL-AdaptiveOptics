```markdown
# Role & Context
你是一个资深的深度学习算法工程师和 Python 架构师。我正在完成一个深度学习课程项目：**“基于生成模型的计算自适应光学：卫星遥感图像去湍流与超分”**。
我们的核心任务是 Image-to-Image Translation（图像到图像翻译），将受大气湍流、光学模糊和噪声影响的“退化遥感图像”恢复为“高清清晰图像”。

# Tech Stack
- 核心框架: PyTorch, Torchvision
- 评估指标: skimage (PSNR, SSIM), lpips
- 前端 Demo: Gradio 或 Streamlit
- 辅助工具: numpy, opencv-python, matplotlib, pyyaml (用于配置管理)

# Project Structure Requirement
请严格按照以下目录结构规划和生成代码：

```text
project_root/
├── configs/                 # 存放配置文件 (如 train_unet.yaml, train_gan.yaml)
├── data/
│   ├── get_data.py          # 下载开源数据集 (如 UC Merced, WHU-RS19 等) 的脚本
│   └── dataset.py           # PyTorch Dataset 和 DataLoader 定义
├── modules/
│   ├── baseline_unet.py     # Baseline 模型: 基础的 U-Net 或 ResNet
│   ├── gan_models.py        # 进阶模型: Pix2Pix 相关的 Generator 和 Discriminator
│   └── diffusion.py         # 核心模型: 轻量级条件扩散模型 (Conditional Diffusion) 骨架
├── utils/
│   ├── degradation.py       # 核心！物理退化模拟器: 包含泽尼克多项式相位屏模拟、PSF模糊、噪声等器
│   ├── metrics.py           # 评估工具: 计算 PSNR, SSIM, LPIPS
│   └── visualization.py     # 可视化工具: 保存成对的图像 (Input, GT, Output)
├── demo/
│   └── app.py               # 基于 Gradio 的交互式 Web Demo
├── train.py                 # 统一的训练脚本 (支持读取 YAML 配置)
├── eval.py                  # 统一的评估脚本
└── requirements.txt         # 项目依赖
```

# Execution Plan (分步执行指南)
请按照以下 Phase 逐一向我确认并生成代码，不要一次性输出所有代码，确保每个模块可独立运行：

**Phase 1: 数据流与物理模拟 (Data & Degradation)**
1. 生成 `utils/degradation.py`：利用 OpenCV/Numpy 或现有的 PyTorch 算子，实现一个函数 `add_atmospheric_turbulence(image)`，利用高斯模糊、高阶随机噪声或模拟的泽尼克多项式（Zernike polynomials）来模拟大气湍流。
2. 生成 `data/dataset.py`：构建 PyTorch 的 `Dataset` 类，能在 `__getitem__` 中读取清晰图像，实时调用 `degradation.py` 生成模糊图像，返回 `(degraded_img, clear_img)` 对。

**Phase 2: 模型构建 (Models)**
1. 生成 `modules/baseline_unet.py`，实现一个标准的 U-Net 作为保底 Baseline。
2. 生成 `modules/gan_models.py`，实现一个带有物理约束的条件 GAN 网络骨架。

**Phase 3: 训练与评估管道 (Training Pipeline)**
1. 生成 `configs/default.yaml` 来管理超参数（learning_rate, batch_size, epochs 等）。
2. 生成 `train.py`：包括标准的 PyTorch 训练循环、Loss 计算（L1 Loss + 评估 metrics 监控）、Optimizer 配置以及模型保存逻辑（保存至 `./checkpoints`）。
3. 生成 `utils/metrics.py`：封装 PSNR, SSIM 和 LPIPS 的计算逻辑。

**Phase 4: 展示与接口 (Demo & Inference)**
1. 生成 `demo/app.py`：使用 Gradio 构建一个优美的 UI，包含一个上传图片的组件，一个“去除湍流”的按钮，以及一个用于对比前后的 Image Slider 组件。

# Coding Constraints (代码规范)
1. 所有的 Python 文件必须包含清晰的 Docstring (模块说明和函数说明)。
2. 必须包含 Type Hints (类型注解)，如 `def forward(self, x: torch.Tensor) -> torch.Tensor:`。
3. 代码必须是 Device-agnostic 的 (支持 `device = 'cuda' if torch.cuda.is_available() else 'cpu'`)。
4. 在涉及复杂物理公式（如相位屏、PSF生成）的地方，务必加上行内注释说明物理含义。
```