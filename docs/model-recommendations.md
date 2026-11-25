# Model Recommendations

[← Back to main README](../README.md)

## Inference speed

Here are the inference speed benchmark results for PlantCaduceus (v1) and PlantCAD2 models:

| Model             | Seq Len | Batch | Peak memory (GB) | Seq/s  | Tokens/s |
| :---------------- | :------ | :---- | :--------------- | :----- | :------- |
| PlantCaduceus_l20 | 512     | 16    | 0.31             | 400.28 | 204,942  |
| PlantCaduceus_l20 | 512     | 32    | 0.56             | 640.86 | 328,118  |
| PlantCaduceus_l20 | 512     | 64    | 1.01             | 663.04 | 339,475  |
| PlantCaduceus_l24 | 512     | 16    | 0.43             | 335.88 | 171,970  |
| PlantCaduceus_l24 | 512     | 32    | 0.75             | 392.83 | 201,140  |
| PlantCaduceus_l24 | 512     | 64    | 1.37             | 407.38 | 208,577  |
| PlantCaduceus_l28 | 512     | 16    | 0.77             | 207.61 | 106,295  |
| PlantCaduceus_l28 | 512     | 32    | 1.27             | 213.99 | 109,563  |
| PlantCaduceus_l28 | 512     | 64    | 2.22             | 219.97 | 112,626  |
| PlantCaduceus_l32 | 512     | 16    | 1.1              | 130.56 | 66,848   |
| PlantCaduceus_l32 | 512     | 32    | 1.71             | 132.62 | 67,902   |
| PlantCaduceus_l32 | 512     | 64    | 2.97             | 135.05 | 69,144   |
| PlantCAD2-Small   | 8192    | 16    | 6.56             | 19.61  | 160,653  |
| PlantCAD2-Small   | 8192    | 32    | 12.76            | 19.26  | 157,767  |
| PlantCAD2-Small   | 8192    | 64    | 24.89            | 19     | 155,649  |
| PlantCAD2-Medium  | 8192    | 16    | 9.62             | 6.88   | 56,386   |
| PlantCAD2-Medium  | 8192    | 32    | 17.62            | 6.76   | 55,342   |
| PlantCAD2-Medium  | 8192    | 64 †  | 33.62            | 6.79   | 55,636   |
| PlantCAD2-Large   | 8192    | 16    | 14.89            | 3.92   | 32,111   |
| PlantCAD2-Large   | 8192    | 32    | 26.95            | 3.87   | 31,741   |
| PlantCAD2-Large   | 8192    | 64 †  | 51.09            | 3.89   | 31,833   |

† Above the [batch size limit](#batch-size-limit-for-plantcad2) for this model and length: the timing is valid, but the model outputs at this batch size are wrong.

## Batch size limit for PlantCAD2

PlantCAD2 models use Mamba2 layers. Current releases of `mamba-ssm` compute memory offsets in these layers with 32-bit integers, which overflow once **batch size × sequence length × the layer's input-projection width** exceeds 2³¹. The sequences past that point in each batch then get wrong results. Sometimes the run stops with `CUDA error: an illegal memory access was encountered`, but often it finishes with no error at all (on GH200 we have only seen silent errors). Training is affected the same way, through wrong outputs and gradients.

Largest safe batch size:

| Model            | 2048 bp | 4096 bp | 8192 bp |
| :--------------- | ------: | ------: | ------: |
| PlantCAD2-Small  | 325     | 162     | 81      |
| PlantCAD2-Medium | 246     | 123     | 61      |
| PlantCAD2-Large  | 165     | 82      | 41      |

The scripts in `src/` enforce this:

- **Inference and embeddings** (`zero_shot_score.py`, `zero_shot_score_sv.py`, `zero-shot-eval.py`, `train_XGBoost.py`, `predict_XGBoost.py`, and `evaluate` / `predict` in `lora_fine_tune.py`) lower the batch size to the limit and log a warning. Results do not depend on the batch size.
- **Training** (`train` in `lora_fine_tune.py`, and `HF_pre_train.py`) stops with an error that suggests a smaller batch size and more gradient-accumulation steps with the same effective batch size.

If you run PlantCAD2 from your own code, keep the batch size at or below this table. PlantCaduceus (v1) uses Mamba1 and is not affected. Upstream: [state-spaces/mamba#1015](https://github.com/state-spaces/mamba/issues/1015) (fix proposed in [#1038](https://github.com/state-spaces/mamba/pull/1038), not yet released).

## Which model to use?

### Variant Effect Analysis (Zero-Shot Scoring)

| Scenario                     | Recommended Model     | `-contextSize` | GPU Memory* | Notes                                                                                   |
| :--------------------------- | :-------------------- | :-------------- | :---------- | :-------------------------------------------------------------------------------------- |
| Limited GPU                  | `PlantCaduceus_l32` | `512` (fixed)  | ~2–3 GB    | Works well for both coding and noncoding variants. Fast and lightweight.                |
| Higher noncoding sensitivity | `PlantCAD2-Medium`  | `≥ 2048`      | ~10–34 GB  | Longer context improves sensitivity for noncoding regions (promoters, enhancers, etc.). |
| Best accuracy                | `PlantCAD2-Large`   | `≥ 2048`      | ~15–51 GB  | Highest accuracy overall. Use `4096` or `8192` if GPU memory allows.                |

\* Approximate peak memory at batch size 16–64. See the inference speed table above for details.

> **How to choose:**
>
> - `PlantCaduceus_l32` with `-contextSize 512` is a strong baseline that works well for both **coding and noncoding** variants, and is fast and lightweight.
> - If you want **higher sensitivity for noncoding regions** (e.g., promoters, enhancers, intergenic variants), use `PlantCAD2-Medium` or `PlantCAD2-Large` with `-contextSize` of at least `2048`. The longer context window helps capture longer-range regulatory signals.
> - For the **best overall accuracy**, use `PlantCAD2-Large` with `-contextSize` of `4096` or `8192` if your GPU memory allows.
>
> **Note:** The script automatically enforces context size constraints — PlantCaduceus is fixed at 512, and PlantCAD2 has a minimum of 2048. You do not need to worry about setting an invalid context size.

### Other Downstream Tasks

- **Long Sequences:** For tasks requiring sequences longer than 512bp, we highly recommend fine-tuning **PlantCAD2** models. PlantCAD (v1) models are limited to a 512bp context window, whereas PlantCAD2 supports up to 8192bp.
