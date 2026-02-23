# visual-forward-model

A collection of models that predict the future state of a robotic manipulation scene given an initial image and an action. Given a 128×128 RGB observation and one of 4 possible actions, the models learn to predict where a target object will end up.

---

## Models

### Deliverable 1 — MLP (`MLP.py`)
A simple fully-connected network. The input image is flattened to a 49,152-length vector, concatenated with a one-hot encoded action, and passed through 4 linear layers to predict the `[X, Y]` position of the object.

### Deliverable 2 — CNN (`CNN.py`)
A convolutional network that extracts spatial features from the image using 4 strided conv layers before fusing with the action and predicting `[X, Y]`. Significantly better at understanding spatial structure than the MLP.

### Deliverable 3 — Image Reconstructor (`reconstructor.py`)
An encoder-decoder (autoencoder-style) network that predicts the **full final image** rather than just coordinates. The encoder compresses the image to an 8×8 feature map, fuses with the action, and the decoder reconstructs the expected future scene pixel by pixel.

---

## Loss Curves

### MLP
![MLP Loss Curve](assets/mlp_loss.png)

### CNN
![CNN Loss Curve](assets/cnn_loss.png)

> Place your loss curve images in an `assets/` folder and name them `mlp_loss.png` and `cnn_loss.png` to render them here.

---

## Dataset

The dataset directory should have the following structure:

```
dataset/
├── init_imgs_0.pt
├── actions_0.pt
├── final_imgs_0.pt
├── init_imgs_1.pt
├── actions_1.pt
├── final_imgs_1.pt
└── ...
```

All datasets are split **80% train / 10% val / 10% test** with a fixed seed, so the test set is always the same across runs.

---

## Usage

### MLP

**Train:**
```bash
python MLP.py train
```

**Train with custom args:**
```bash
python MLP.py train \
  --data_dir dataset \
  --batch_size 128 \
  --num_epochs 50 \
  --lr 0.0001 \
  --checkpoint_dir checkpoints_MLP \
  --logs_file training_logs_MLP.csv
```

**Test:**
```bash
python MLP.py test
```

**Test with custom checkpoint:**
```bash
python MLP.py test \
  --checkpoint_path checkpoints_MLP/best_model.pt \
  --data_dir dataset
```

| Argument | Default | Description |
|---|---|---|
| `mode` | — | `train` or `test` (required) |
| `--data_dir` | `dataset` | Path to dataset directory |
| `--batch_size` | `128` | Batch size |
| `--num_epochs` | `50` | Number of training epochs |
| `--lr` | `0.0001` | Learning rate |
| `--checkpoint_dir` | `checkpoints_MLP` | Directory to save checkpoints |
| `--logs_file` | `training_logs_MLP.csv` | Path to training logs CSV |
| `--checkpoint_path` | `checkpoints_MLP/best_model.pt` | Checkpoint to load for testing |

---

### CNN

**Train:**
```bash
python CNN.py train
```

**Train with custom args:**
```bash
python CNN.py train \
  --data_dir dataset \
  --batch_size 128 \
  --num_epochs 100 \
  --lr 0.0001 \
  --checkpoint_dir checkpoints_CNN \
  --logs_file training_logs_CNN.csv
```

**Test:**
```bash
python CNN.py test
```

**Test with custom checkpoint:**
```bash
python CNN.py test \
  --checkpoint_path checkpoints_CNN/best_model.pt \
  --data_dir dataset \
  --out_dir reconstructed_images \
  --num_test_visuals 20
```

| Argument | Default | Description |
|---|---|---|
| `mode` | — | `train` or `test` (required) |
| `--data_dir` | `dataset` | Path to dataset directory |
| `--batch_size` | `128` | Batch size |
| `--num_epochs` | `100` | Number of training epochs |
| `--lr` | `0.0001` | Learning rate |
| `--checkpoint_dir` | `checkpoints_CNN` | Directory to save checkpoints |
| `--logs_file` | `training_logs_CNN.csv` | Path to training logs CSV |
| `--checkpoint_path` | `checkpoints_CNN/best_model.pt` | Checkpoint to load for testing |
| `--out_dir` | `reconstructed_images` | Directory to save visual comparisons |
| `--num_test_visuals` | `20` | Number of comparison images to save |

---

### Image Reconstructor

**Train:**
```bash
python reconstructor.py train
```

**Train with custom args:**
```bash
python reconstructor.py train \
  --data_dir dataset \
  --batch_size 32 \
  --num_epochs 50 \
  --lr 0.0005 \
  --checkpoint_path reconstructor_model.pt \
  --logs_file training_logs_reconstructor.csv
```

**Test:**
```bash
python reconstructor.py test
```

**Test with custom args:**
```bash
python reconstructor.py test \
  --checkpoint_path reconstructor_model.pt \
  --comparison_dir test_comparisons \
  --num_comparison_imgs 16
```

| Argument | Default | Description |
|---|---|---|
| `mode` | — | `train` or `test` (required) |
| `--data_dir` | `dataset` | Path to dataset directory |
| `--batch_size` | `32` | Batch size |
| `--num_epochs` | `50` | Number of training epochs |
| `--lr` | `0.0005` | Learning rate |
| `--checkpoint_path` | `reconstructor_model.pt` | Path to save/load model checkpoint |
| `--logs_file` | `training_logs_reconstructor.csv` | Path to training logs CSV |
| `--comparison_dir` | `test_comparisons` | Directory to save per-sample comparison images |
| `--num_comparison_imgs` | `16` | Number of samples to save as comparison images |

---

## Requirements

```bash
pip install torch torchvision pillow
```
