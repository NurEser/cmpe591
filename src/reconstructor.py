import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.utils as vutils
import os
import argparse
import csv
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

torch.manual_seed(42)


import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetReconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- ENCODER ---
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())

        # --- ACTION INJECTION (No Flattening!) ---
        self.action_mlp = nn.Sequential(nn.Linear(4, 128), nn.ReLU())

        # --- DECODER ---
        # Note the channel math: Decoder output + Skip Connection input
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU())
        
        # 64 channels from dec4 + 64 channels from enc3 = 128 channels in
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1), nn.ReLU())
        
        # 32 channels from dec3 + 32 channels from enc2 = 64 channels in
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1), nn.ReLU())
        
        # 16 channels from dec2 + 16 channels from enc1 = 32 channels in
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid())

    def forward(self, img, action_id):
        # 1. Encode and save the spatial features for later
        e1 = self.enc1(img)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 2. Inject the action directly into the bottleneck 
        a_onehot = F.one_hot(action_id, num_classes=4).float()
        a_emb = self.action_mlp(a_onehot).view(-1, 128, 1, 1)
        bottleneck = e4 + a_emb # Broadcasts the action across the 8x8 spatial grid

        # 3. Decode using Skip Connections (torch.cat)
        d4 = self.dec4(bottleneck)
        d3 = self.dec3(torch.cat([d4, e3], dim=1)) # Bridge 3
        d2 = self.dec2(torch.cat([d3, e2], dim=1)) # Bridge 2
        out = self.dec1(torch.cat([d2, e1], dim=1)) # Bridge 1
        
        return out
    
# --- 2. DATA LOADING ---
def load_and_prep_data(data_dir):
    print("Loading data files...")
    in_imgs_list, actions_list, out_imgs_list = [], [], []

    for file in os.listdir(data_dir):
        if file.startswith("init_imgs"):
            idx = file.split("_")[-1].split(".")[0]
            in_imgs_list.append(torch.load(os.path.join(data_dir, f"init_imgs_{idx}.pt")))
            actions_list.append(torch.load(os.path.join(data_dir, f"actions_{idx}.pt")))
            out_imgs_list.append(torch.load(os.path.join(data_dir, f"final_imgs_{idx}.pt")))

    in_imgs = torch.cat(in_imgs_list).float() / 255.0
    actions = torch.cat(actions_list).long()
    out_imgs = torch.cat(out_imgs_list).float() / 255.0

    print(f"Total dataset size: {len(in_imgs)} samples.")
    return in_imgs, actions, out_imgs


def split_dataset(full_dataset):
    """Splits dataset into 80% train, 10% val, 10% test consistently."""
    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    return random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
import torch
import torch.nn.functional as F
from math import exp


# --- 2. The Combined Loss Function ---
def hyper_focused_l1_loss(pred, target, red_weight=500.0):
    """L1 Loss that makes ignoring or hallucinating the red cube mathematically agonizing."""
    # 1. Base Pixel Error (Mean Absolute Error)
    base_loss = torch.abs(pred - target)
    
    # 2. Extract Redness (Shape: [Batch, 1, Height, Width])
    t_red = (target[:, 0:1] - torch.max(target[:, 1:2], target[:, 2:3])).clamp(0, 1)
    p_red = (pred[:, 0:1] - torch.max(pred[:, 1:2], pred[:, 2:3])).clamp(0, 1)
    
    # 3. Create a mask of where the cube IS and where the network GUESSED it is
    combined_mask = torch.max(t_red, p_red)
    
    # 4. Apply a penalty multiplier (500x) to the cube pixels
    # The background and robot arm just get the standard 1.0 weight.
    weight_map = 1.0 + (combined_mask * red_weight)
    
    return (weight_map * base_loss).mean()

# --- 3. TRAINING ---
def train(args):
    device = get_device()
    print(f"Training on device: {device}")

    in_imgs, actions, out_imgs = load_and_prep_data(args.data_dir)
    full_dataset = TensorDataset(in_imgs, actions, out_imgs)
    train_dataset, val_dataset, _ = split_dataset(full_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} samples")

    os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)

    with open(args.logs_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train MSE', 'Val MSE'])

    model = UNetReconstructor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    #criterion = nn.MSELoss()
    criterion = lambda pred, target: hyper_focused_l1_loss(pred, target, red_weight=500.0)

    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for b_in, b_act, b_out in train_loader:
            b_in, b_act, b_out = b_in.to(device), b_act.to(device), b_out.to(device)
            optimizer.zero_grad()
            loss = criterion(model(b_in, b_act), b_out)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_in, b_act, b_out in val_loader:
                b_in, b_act, b_out = b_in.to(device), b_act.to(device), b_out.to(device)
                val_loss += criterion(model(b_in, b_act), b_out).item()

        avg_val_loss = val_loss / len(val_loader)

        with open(args.logs_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, avg_train_loss, avg_val_loss])

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"Best model saved: {args.checkpoint_path}")

    print("Training complete!")


# --- 4. TESTING ---
def test(args):
    device = get_device()
    print(f"Testing on device: {device}")

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        return

    model = UNetReconstructor().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.checkpoint_path}")

    print("\nLoading test data...")
    in_imgs, actions, out_imgs = load_and_prep_data(args.data_dir)
    full_dataset = TensorDataset(in_imgs, actions, out_imgs)
    _, _, test_dataset = split_dataset(full_dataset)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating on {len(test_dataset)} test samples...")

    os.makedirs(args.comparison_dir, exist_ok=True)

    test_loss = 0.0
    saved = 0
    all_inputs, all_targets, all_preds = [], [], []

    with torch.no_grad():
        for b_in, b_act, b_out in test_loader:
            b_in, b_act, b_out = b_in.to(device), b_act.to(device), b_out.to(device)
            preds = model(b_in, b_act)
            test_loss += F.mse_loss(preds, b_out).item()

            if saved < args.num_comparison_imgs:
                remaining = args.num_comparison_imgs - saved
                all_inputs.append(b_in[:remaining].cpu())
                all_targets.append(b_out[:remaining].cpu())
                all_preds.append(preds[:remaining].cpu())
                saved += min(len(b_in), remaining)

    avg_test_mse = test_loss / len(test_loader)
    print("\n" + "=" * 50)
    print(f"Test MSE Loss: {avg_test_mse:.6f}")
    print("=" * 50)

    # Save each sample as its own labeled image
    all_inputs = torch.cat(all_inputs)
    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)

    label_height = 24  # pixels reserved for text label above each image

    for i in range(len(all_inputs)):
        gt_pil   = TF.to_pil_image(all_targets[i])   # ground truth
        pred_pil = TF.to_pil_image(all_preds[i])     # prediction
        w, h = gt_pil.size

        canvas = Image.new("RGB", (w * 2, h + label_height), color=(30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        canvas.paste(gt_pil,   (0,     label_height))
        canvas.paste(pred_pil, (w,     label_height))

        # Draw labels
        draw.rectangle([0, 0, w - 1, label_height - 1],     fill=(50, 50, 50))
        draw.rectangle([w, 0, w * 2 - 1, label_height - 1], fill=(50, 50, 50))
        draw.text((8, 4),     "Ground Truth", fill=(255, 255, 255))
        draw.text((w + 8, 4), "Prediction",   fill=(255, 200, 100))

        out_path = os.path.join(args.comparison_dir, f"sample_{i+1:03d}.png")
        canvas.save(out_path)

    print(f"Saved {len(all_inputs)} comparison images to: {args.comparison_dir}")
    print("Test completed successfully!")

# --- 5. MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Train or test the ImageReconstructor model.")

    parser.add_argument("mode", choices=["train", "test"], help="Run mode: 'train' or 'test'")

    # Shared args
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    # Training args
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")

    # Checkpoint / output args
    parser.add_argument("--checkpoint_path", type=str, default="reconstructor_model_unet_final.pt", help="Path to save/load model checkpoint")
    parser.add_argument("--logs_file", type=str, default="training_logs_reconstructor_unet_final.csv", help="Path to training logs CSV")
    parser.add_argument("--comparison_dir", type=str, default="test_comparisons_unet_final", help="Directory to save per-sample comparison images")
    parser.add_argument("--num_comparison_imgs", type=int, default=50, help="Number of samples to save as comparison images")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
