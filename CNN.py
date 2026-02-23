import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import csv
import argparse

class PositionPredictorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- Phase 1: Convolutional Feature Extractor ---
        # nn.Conv2d parameters: (in_channels, out_channels, kernel_size, stride, padding)
        
        # Input: 3 channels (RGB), 128x128 image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  
        # Output after conv1: 16 channels, 64x64 spatial size
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) 
        # Output after conv2: 32 channels, 32x32 spatial size
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) 
        # Output after conv3: 64 channels, 16x16 spatial size
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # Output after conv4: 128 channels, 8x8 spatial size
        
        # Calculate the size of the flattened features
        # 128 channels * 8 width * 8 height
        self.flattened_size = 128 * 8 * 8  # 8192
        
        # --- Phase 2: The Decision Head (MLP) ---
        action_size = 4
        input_dim = self.flattened_size + action_size  # 8196
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # Final Output: X, Y coordinates

    def forward(self, image_state, action_id):
        # 1. Extract Visual Features
        x = F.relu((self.conv1(image_state)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu(self.conv4(x))
        
        # 2. Flatten the Spatial Features
        # (Batch, 128, 8, 8) -> (Batch, 8192)
        flat_features = torch.flatten(x, start_dim=1)
        
        # 3. Fuse with the Action Signal
        encoded_action = F.one_hot(action_id, num_classes=4).float()
        fused_signal = torch.cat((flat_features, encoded_action), dim=1)
        
        # 4. Predict the Coordinates via the Dense Head
        x = F.relu((self.fc1(fused_signal)))
        x = F.relu(self.fc2(x))
        predicted_position = self.fc3(x)
        
        return predicted_position
    
    
  
# --- Data Loading ---
def load_and_prep_data(data_dir):
    print("Loading data files...")
    init_imgs_list, actions_list, positions_list = [], [], []

    for file in os.listdir(data_dir):
        if file.startswith("init_imgs"):
            idx = file.split("_")[-1].split(".")[0]
            init_imgs_list.append(torch.load(os.path.join(data_dir, f"init_imgs_{idx}.pt")))
            actions_list.append(torch.load(os.path.join(data_dir, f"actions_{idx}.pt")))
            positions_list.append(torch.load(os.path.join(data_dir, f"positions_{idx}.pt")))

    all_init_imgs = torch.cat(init_imgs_list, dim=0)
    all_actions = torch.cat(actions_list, dim=0).long()
    all_positions = torch.cat(positions_list, dim=0)
    all_init_imgs = all_init_imgs.float() / 255.0

    print(f"Total dataset size: {len(all_init_imgs)} samples.")
    return all_init_imgs, all_actions, all_positions


def split_dataset(full_dataset):
    """Splits dataset into 80% train, 10% val, 10% test consistently."""
    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    return random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # ensures same split every run
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# --- Training ---
def train(args):
    images, actions, targets = load_and_prep_data(args.data_dir)

    full_dataset = TensorDataset(images, actions, targets)
    train_dataset, val_dataset, _ = split_dataset(full_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print(f"Training on device: {device}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} samples")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(args.logs_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train MSE', 'Val MSE'])

    model = PositionPredictorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_imgs, batch_acts, batch_targs in train_loader:
            batch_imgs, batch_acts, batch_targs = (
                batch_imgs.to(device), batch_acts.to(device), batch_targs.to(device)
            )
            optimizer.zero_grad()
            loss = criterion(model(batch_imgs, batch_acts), batch_targs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_imgs, batch_acts, batch_targs in val_loader:
                batch_imgs, batch_acts, batch_targs = (
                    batch_imgs.to(device), batch_acts.to(device), batch_targs.to(device)
                )
                val_loss += criterion(model(batch_imgs, batch_acts), batch_targs).item()

        avg_val_loss = val_loss / len(val_loader)

        with open(args.logs_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, avg_train_loss, avg_val_loss])

        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

        #save the model if only the validation loss improved
        if epoch == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Best model updated and saved: {best_checkpoint_path}")

    print("Training complete!")


# --- Testing ---
def test(args):
    device = get_device()
    print(f"Testing on device: {device}")

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        return

    model = PositionPredictorCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.checkpoint_path}")

    print("\nLoading test data...")
    images, actions, targets = load_and_prep_data(args.data_dir)

    full_dataset = TensorDataset(images, actions, targets)
    _, _, test_dataset = split_dataset(full_dataset)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = nn.MSELoss()

    print(f"Evaluating on {len(test_dataset)} test samples...")
    total_loss = 0.0
    with torch.no_grad():
        for batch_imgs, batch_acts, batch_targs in test_loader:
            batch_imgs, batch_acts, batch_targs = (
                batch_imgs.to(device), batch_acts.to(device), batch_targs.to(device)
            )
            total_loss += criterion(model(batch_imgs, batch_acts), batch_targs).item()

    avg_test_mse = total_loss / len(test_loader)
    print("\n" + "=" * 50)
    print(f"Test MSE Loss: {avg_test_mse:.6f}")
    print("=" * 50)
    print("Test completed successfully!")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Train or test the PositionPredictorCNN model.")

    parser.add_argument("mode", choices=["train", "test"], help="Run mode: 'train' or 'test'")

    # Shared args
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")

    # Training args
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_CNN", help="Directory to save checkpoints")
    parser.add_argument("--logs_file", type=str, default="training_logs_CNN.csv", help="Path to training logs CSV")

    # Testing args
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=r"checkpoints_CNN/best_model.pt",
        help="Path to checkpoint file for testing"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()