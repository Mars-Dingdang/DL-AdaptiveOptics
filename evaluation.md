对于每个 x=[50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], 我们在 configs/default.yaml 中设置 turbulence_strength=x/100.0，生成一个新的评测数据集，并使用训练好的 UNet 模型进行评测，记录 PSNR 和 SSIM 结果。最后，我们将结果可视化成图表，展示不同 turbulence_strength 下的模型性能变化趋势。

比如 x=55, 我们会在 configs/default.yaml 中设置 turbulence_strength=0.55，然后运行以下命令：

```bash
source .venv/bin/activate
python data/build_seven_frame_eval.py \
	--config configs/default.yaml \
	--input-root data/clean_patches/nwpu_parquet/images \
	--output-lmdb-root data/eval_seven_frame_mild55 \
	--viz-dir outputs/eval_seven_frame_degradation_mild55 \
	--start-index 2000 \
	--count 10 \
	--num-frames 7 \
	--force

python eval.py \
	--config configs/default.yaml \
	--checkpoint checkpoints/best_unet.pt \
	--split test \
	--test-root data/eval_seven_frame_mild55 \
	--batch-size 1 \
	--num-workers 0 \
	--out-dir outputs/eval_unet_seven_frame_mild55 \
	--sample-dir outputs/eval_unet_seven_frame_mild55 \
	--metrics-path outputs/eval_unet/metric_seven_frame_mild55.txt \
	--save-images \
	--max-save 10

python data/build_single_frame_eval.py \
	--config configs/default.yaml \
	--input-root data/clean_patches/nwpu_parquet/images \
	--output-lmdb-root data/eval_single_frame_mild55 \
	--viz-dir outputs/eval_single_frame_degradation_mild55 \
	--start-index 2000 \
	--count 10 \
	--num-frames 7 \
	--force

python eval.py \
	--config configs/default.yaml \
	--checkpoint checkpoints/best_unet.pt \
	--split test \
	--test-root data/eval_single_frame_mild55 \
	--batch-size 1 \
	--num-workers 0 \
	--out-dir outputs/eval_unet_single_frame_mild55 \
	--sample-dir outputs/eval_unet_single_frame_mild55 \
	--metrics-path outputs/eval_unet/metric_single_frame_mild55.txt \
	--save-images \
	--max-save 10
```

最后我希望把所有eval的数据整理在一起，并生成两个折线图：第一个折线图展示 turbulence_strength 从 0.5 到 1.0 的变化对PSNR 指标的影响（两条折线分别展示单帧和7帧）；第二个折线图展示 turbulence_strength 从 0.5 到 1.0 的变化对SSIM 指标的影响（两条折线分别展示单帧和7帧）。

```bash
source .venv/bin/activate
python eval_auto.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best_unet.pt \
  --output-root outputs/eval_strength_sweep_unet \
  --count 10 \
  --start-index 2000 \
  --num-frames 7 \
  --batch-size 1 \
  --num-workers 0
```

如果要对新的 mild75 模型 checkpoints/mild75/best_unet.pt 做同样的泛化实验，但直接复用已经生成好的评测数据集 data/eval_seven_frame_mild{50..100} 和 data/eval_single_frame_mild{50..100}，可以使用下面的命令：

```bash
source .venv/bin/activate
python eval_auto.py \
	--config configs/default.yaml \
	--checkpoint checkpoints/mild75/best_unet.pt \
	--output-root outputs/generalization_75 \
	--batch-size 1 \
	--num-workers 0 \
	--max-save 10 \
	--reuse-existing-datasets
```

这条命令不会重新执行 degradation，而是直接评测现有的 single-frame 和 seven-frame mild50~mild100 数据集，并将结果统一输出到 outputs/generalization_75 下：

- 单帧样例图与评测输出位于 outputs/generalization_75/single_frame/mildXX/
- 7 帧样例图与评测输出位于 outputs/generalization_75/seven_frame/mildXX/
- 指标文本位于 outputs/generalization_75/metrics/
- 汇总 CSV 与折线图位于 outputs/generalization_75/summary/