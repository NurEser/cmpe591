# CMPE591 - HW1

A collection of models that predict the future state of a robotic manipulation scene given an initial image and an action. Given a 128×128 RGB observation and one of 4 possible actions, the models learn to predict where a target object will end up.

---

## Models

### Deliverable 1 — MLP (`MLP.py`)
A simple fully-connected network. The input image is flattened to a 49,152-length vector, concatenated with a one-hot encoded action, and passed through 4 linear layers to predict the `[X, Y]` position of the object.

### Deliverable 2 — CNN (`CNN.py`)
A convolutional network that extracts spatial features from the image using 4 strided conv layers before fusing with the action and predicting `[X, Y]`. Significantly better at understanding spatial structure than the MLP.

### Deliverable 3 — Image Reconstructor (`reconstructor.py`)
 UNet-style encoder-decoder with skip connections that predicts the **full final image** rather than just coordinates. The encoder compresses the image through 4 strided conv layers, the action is injected at the bottleneck via a small MLP and added directly to the feature map, and the decoder reconstructs the expected future scene using skip connections from each encoder stage. Trained with a red-object-focused weighted L1 loss and a learning rate scheduler.

---

## Loss Curves

### MLP

<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/1e45ee66-963c-4d74-a8d3-a4510053074f" />

--- Training Summary ---
Best Train MSE: 0.050100 at epoch 46
Best Val MSE: 0.048612 at epoch 44
Final Train MSE: 0.052131
Final Val MSE: 0.053684

Test MSE Loss: 0.051761
==================================================

### CNN

<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/7a73c024-16cb-4740-b2da-eee9bbe81d45" />

--- Training Summary ---
Best Train MSE: 0.007812 at epoch 99
Best Val MSE: 0.006240 at epoch 100
Final Train MSE: 0.007857
Final Val MSE: 0.006240

Test MSE Loss: 0.007329
==================================================



### UNET

<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/abc35b23-cee9-4017-a1be-16325d61592d" />

--- Training Summary ---
Best Train MSE: 0.149860 at epoch 122
Best Val MSE: 0.149002 at epoch 123
Final Train MSE: 0.149866
Final Val MSE: 0.149002

Test MSE Loss: 0.004317
==================================================

---

## Reconstructed Images

<img width="256" height="152" alt="image" src="https://github.com/user-attachments/assets/0e00941d-4b32-4087-8768-50d933ec8bd1" />


<img width="256" height="152" alt="image" src="https://github.com/user-attachments/assets/5417bcdc-3640-4706-a8e5-4ac8cbdc94d1" />

<img width="256" height="152" alt="image" src="https://github.com/user-attachments/assets/417b2ad8-68dd-4470-98f6-d59e917f7f10" />

<img width="256" height="152" alt="image" src="https://github.com/user-attachments/assets/cda79675-a057-4c8a-85b9-ee2ad14f1dec" />

<img width="256" height="152" alt="image" src="https://github.com/user-attachments/assets/5a49e15f-55ae-472b-b2d5-df3dec3b0c76" />

<img width="256" height="152" alt="image" src="https://github.com/user-attachments/assets/a94b9483-2b2b-4c16-9af8-dae6a30d80b5" />

## Test Errors

### MLP


Loading test data...
Loading data files...
Total dataset size: 1000 samples.
Evaluating on 100 test samples...


Test completed successfully!

### CNN
Loading test data...
Loading data files...
Total dataset size: 1000 samples.
Evaluating on 100 test samples...


Test completed successfully!


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

### Image Reconstructor (UNet)

**Train:**
```bash
python reconstructor.py train
```

**Train with custom args:**
```bash
python reconstructor.py train \
  --data_dir dataset \
  --batch_size 32 \
  --num_epochs 200 \
  --lr 0.0005 \
  --checkpoint_path reconstructor_model_unet_final.pt \
  --logs_file training_logs_reconstructor_unet_final.csv
```

**Test:**
```bash
python reconstructor.py test
```

**Test with custom args:**
```bash
python reconstructor.py test \
  --checkpoint_path reconstructor_model_unet_final.pt \
  --comparison_dir test_comparisons_unet_final \
  --num_comparison_imgs 50
```

| Argument | Default | Description |
|---|---|---|
| `mode` | — | `train` or `test` (required) |
| `--data_dir` | `dataset` | Path to dataset directory |
| `--batch_size` | `32` | Batch size |
| `--num_epochs` | `200` | Number of training epochs |
| `--lr` | `0.0005` | Learning rate |
| `--checkpoint_path` | `reconstructor_model_unet_final.pt` | Path to save/load model checkpoint |
| `--logs_file` | `training_logs_reconstructor_unet_final.csv` | Path to training logs CSV |
| `--comparison_dir` | `test_comparisons_unet_final` | Directory to save per-sample comparison images |
| `--num_comparison_imgs` | `50` | Number of samples to save as comparison images |

---

## Requirements

```bash
pip install -r requirements.txt
```
