# XGBoost Classifiers

[← Back to main README](../README.md)

## Training XGBoost classifiers

Train custom classifiers on top of PlantCAD embeddings for specific annotation tasks (e.g., TIS, TTS, splice sites).

**Purpose**: Fine-tune prediction performance for specific annotation tasks using supervised learning.

**Data format**: Training data should follow the format used in our [cross-species annotation dataset](https://huggingface.co/datasets/kuleshov-group/cross-species-single-nucleotide-annotation/tree/main/TIS).

```bash
python src/train_XGBoost.py \
    -train train.tsv \
    -valid valid.tsv \
    -test test_rice.tsv \
    -model 'kuleshov-group/PlantCaduceus_l20' \
    -output ./output \
    -device 'cuda:0'
```

**Expected outputs:**

- Trained XGBoost classifier (`.json` file)
- Performance metrics on validation/test sets
- Feature importance analysis

## Using pre-trained XGBoost classifiers

We provide pre-trained XGBoost classifiers for common annotation tasks in the [`classifiers`](../classifiers) directory.

**Available classifiers:**

- TIS (Translation Initiation Sites)
- TTS (Translation Termination Sites)
- Splice donor/acceptor sites

```bash
python src/predict_XGBoost.py \
    -test test_rice.tsv \
    -model 'kuleshov-group/PlantCaduceus_l20' \
    -classifier classifiers/PlantCaduceus_l20/TIS_XGBoost.json \
    -device 'cuda:0' \
    -output ./output
```

**Expected output**: Predictions with confidence scores for each sequence in your test data.

## GPU memory

Embedding extraction runs the model with `output_hidden_states=True`, so memory grows fast with `-batchSize` (default 128 in both scripts). With `PlantCaduceus_l32` (weights loaded in float32), peak memory is about 3.5 GB at `-batchSize 8`, 6.2 GB at 16 and 11.4 GB at 32, measured on an RTX 2070 SUPER. On an 8 GB GPU, use `-batchSize 8`.

Running out of GPU memory here can show up as `CUBLAS_STATUS_EXECUTION_FAILED` instead of an out-of-memory error.
