Evaluation实验：
1. 单帧输入的结果：从 data/clean_patches/nwpu_parquet/images 的 2000 开始，将 2000～2029 的 30 张 clean 图先退化成单帧大气湍流图，再把该单帧复制为 7 帧相同输入，构造成兼容现有 sequence UNet 的 evaluation set，存到 data/eval_single_frame/data.mdb 和 data/eval_single_frame/lock.mdb 中。

实现说明：
- 为避免覆盖已有 outputs/eval_unet/samples，本次 degradation 可视化改为输出到独立目录 outputs/eval_single_frame_degradation，命名仍保持 unet_00000.png ~ unet_00029.png。
- 评测指标按原需求输出到 outputs/eval_unet/metric_single_frame.txt。
- UNet 恢复结果输出到 outputs/eval_unet_single_frame。

可复现命令：

1. 构建单帧 evaluation LMDB 与 degradation 预览图

```bash
source .venv/bin/activate
python data/build_single_frame_eval.py \
	--config configs/default.yaml \
	--input-root data/clean_patches/nwpu_parquet/images \
	--output-lmdb-root data/eval_single_frame_mild75 \
	--viz-dir outputs/eval_single_frame_degradation_mild75 \
	--start-index 2000 \
	--count 10 \
	--num-frames 7 \
	--force
```

如果当前环境没有可用 GPU，可以临时改用较慢但更稳的 CPU 或快速近似后端：

```bash
python data/build_single_frame_eval.py \
	--config configs/default.yaml \
	--input-root data/clean_patches/nwpu_parquet/images \
	--output-lmdb-root data/eval_single_frame \
	--viz-dir outputs/eval_single_frame_degradation \
	--start-index 2000 \
	--count 30 \
	--num-frames 7 \
	--backend turbsim_cpu_v1 \
	--force
```

2. 用现有 UNet checkpoint 做单帧评测

```bash
source .venv/bin/activate
python eval.py \
	--config configs/default.yaml \
	--checkpoint checkpoints/best_unet.pt \
	--split test \
	--test-root data/eval_single_frame_mild75 \
	--batch-size 1 \
	--num-workers 0 \
	--out-dir outputs/eval_unet_single_frame_mild75 \
	--sample-dir outputs/eval_unet_single_frame_mild75 \
	--metrics-path outputs/eval_unet/metric_single_frame_mild75.txt \
	--save-images \
	--max-save 10
```

3. 快速检查产物数量

```bash
ls outputs/eval_single_frame_degradation | wc -l
ls outputs/eval_unet_single_frame | wc -l
cat outputs/eval_unet/metric_single_frame.txt
```

4. 七帧更强 degradation 的结果：从 data/clean_patches/nwpu_parquet/images 的 2000 开始，将 2000～2029 的 30 张 clean 图直接退化成更强的 7 帧 turbulence sequence，构造成新的 evaluation set，存到 data/eval_seven_frame/data.mdb 和 data/eval_seven_frame/lock.mdb 中，并输出每个样本的 7 帧 degradation 接触图到 outputs/eval_seven_frame_degradation。

实现说明：
- 该实验使用新的 data/build_seven_frame_eval.py。
- 它不再复制单帧，而是直接生成 7 帧真实 sequence。
- 脚本内部会强制使用更强的退化设置：关闭 luma_only、关闭 reuse_psf_per_frame、恢复更高 PSF 分辨率，并提高 turbulence_strength / cn2 / wind speed 的有效强度。
- 模型评测仍然使用现有 checkpoints/best_unet.pt，通过 eval.py 直接读取 data/eval_seven_frame。

可复现命令：

1. 构建七帧强退化 evaluation LMDB 与 degradation 预览图

```bash
source .venv/bin/activate
python data/build_seven_frame_eval.py \
	--config configs/default.yaml \
	--input-root data/clean_patches/nwpu_parquet/images \
	--output-lmdb-root data/eval_seven_frame_mild75 \
	--viz-dir outputs/eval_seven_frame_degradation_mild75 \
	--start-index 2000 \
	--count 10 \
	--num-frames 7 \
	--force
```

如果当前环境没有可用 GPU，可以切到 CPU 后端：

```bash
python data/build_seven_frame_eval.py \
	--config configs/default.yaml \
	--input-root data/clean_patches/nwpu_parquet/images \
	--output-lmdb-root data/eval_seven_frame_mild100 \
	--viz-dir outputs/eval_seven_frame_degradation_mild100 \
	--start-index 2000 \
	--count 10 \
	--num-frames 7 \
	--backend turbsim_cpu_v1 \
	--force
```

2. 用现有 UNet checkpoint 做七帧强退化评测

```bash
source .venv/bin/activate
python eval.py \
	--config configs/default.yaml \
	--checkpoint checkpoints/best_unet.pt \
	--split test \
	--test-root data/eval_seven_frame_mild75 \
	--batch-size 1 \
	--num-workers 0 \
	--out-dir outputs/eval_unet_seven_frame_mild75 \
	--sample-dir outputs/eval_unet_seven_frame_mild75 \
	--metrics-path outputs/eval_unet/metric_seven_frame_mild75.txt \
	--save-images \
	--max-save 10
```

3. 快速检查产物数量

```bash
ls outputs/eval_seven_frame_degradation_mild100 | wc -l
ls outputs/eval_unet_seven_frame_mild100 | wc -l
cat outputs/eval_unet/metric_seven_frame_mild100.txt
```