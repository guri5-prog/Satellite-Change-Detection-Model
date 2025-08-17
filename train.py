import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
from datetime import datetime

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as sk_f1_score, jaccard_score
except ImportError:
    print("[⚠] scikit-learn not found. Installing...")
    os.system(f"{sys.executable} -m pip install scikit-learn")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as sk_f1_score, jaccard_score

from models.changenet import ChangeNet
from dataset.change_dataset import ChangeDataset
from utils.losses import BCEDiceLoss
# --- Import your custom metrics ---
from utils.metrics import iou_score, f1_score
from utils.checkpoint import save_checkpoint, load_checkpoint


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_model(model, val_loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", ncols=100):
            a = batch['a'].to(device)
            b = batch['b'].to(device)
            mask = batch['mask'].to(device)

            # Get raw logits from the model
            logits = model(a, b)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = mask.cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    return {
        "probs": np.concatenate(all_probs, axis=0),
        "labels": np.concatenate(all_labels, axis=0)
    }


def train():
    config = load_config("configs/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[🖥] Training on device: {device}")
    if torch.cuda.is_available():
        print(f"   🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠ Running on CPU - training will be slower.")

    dataset_root = config['dataset_path']
    image_size = config['image_size']

    print("\n📂 Loading TRAIN dataset...")
    train_dataset = ChangeDataset(
        root_dir=dataset_root,
        split='train',
        image_size=image_size,
        use_augmentations=config['augmentations']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    print("\n📂 Loading VAL dataset...")
    val_dataset = ChangeDataset(
        root_dir=dataset_root,
        split='val',
        image_size=image_size,
        use_augmentations=False
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    model = ChangeNet(backbone=config['backbone']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = BCEDiceLoss()

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resume training if last checkpoint exists
    start_epoch, best_f1 = 0, 0.0
    last_ckpt_path = os.path.join(ckpt_dir, "last_checkpoint.pth")
    if os.path.exists(last_ckpt_path):
        print(f"\n[🔄] Resuming training from last checkpoint: {last_ckpt_path}")
        start_epoch, last_loss, best_f1 = load_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, device
        )
        print(f"[📈] Resumed from epoch {start_epoch} | Last loss: {last_loss:.4f} | Best F1: {best_f1:.4f}")

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        running_loss, running_iou, running_f1 = 0.0, 0.0, 0.0
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['epochs']}] LR: {current_lr:.1e}", ncols=120)
        for i, batch in enumerate(progress_bar):
            a, b, mask = batch['a'].to(device), batch['b'].to(device), batch['mask'].to(device)

            # Forward pass to get raw logits
            logits = model(a, b)
            loss = criterion(logits, mask)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics using the custom functions
            # These functions should handle the sigmoid activation internally
            iou = iou_score(logits, mask)
            f1 = f1_score(logits, mask)

            running_loss += loss.item()
            running_iou += iou
            running_f1 += f1

            progress_bar.set_postfix({
                "Loss": f"{running_loss / (i + 1):.4f}",
                "IoU": f"{running_iou / (i + 1):.4f}",
                "F1": f"{running_f1 / (i + 1):.4f}"
            })

        scheduler.step()
        avg_loss_epoch = running_loss / len(train_loader)
        avg_f1_epoch = running_f1 / len(train_loader)

        # Save last checkpoint (overwrites each time)
        save_checkpoint(model, optimizer, scheduler, epoch, avg_loss_epoch, best_f1, ckpt_dir, "last_checkpoint.pth")

        # Save best model if F1 improves
        if avg_f1_epoch > best_f1:
            best_f1 = avg_f1_epoch
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss_epoch, best_f1, ckpt_dir, "best_model.pth")
            print(f"\n⭐ New best F1: {best_f1:.4f} → Saved as best_model.pth")

    print("\n🔍 Evaluating best model on validation set...")
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        # Load the state_dict from the best model checkpoint
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])

    metrics = evaluate_model(model, val_loader, device)
    all_probs, all_labels = metrics["probs"], metrics["labels"]

    # Find the best threshold on the validation set
    best_threshold, best_f1_score_val = 0.5, 0
    print("\n🛠 Finding best threshold...")
    for threshold in np.arange(0.1, 0.91, 0.05):
        binary_preds = (all_probs > threshold).astype(np.uint8)
        f1 = sk_f1_score(all_labels.flatten(), binary_preds.flatten())
        if f1 > best_f1_score_val:
            best_f1_score_val, best_threshold = f1, threshold
            
    print(f"\n✅ Best Threshold found: {best_threshold:.2f} (achieved F1: {best_f1_score_val:.4f})")
    
    # Calculate final metrics using the best threshold
    final_preds = (all_probs > best_threshold).astype(np.uint8)
    print("\n📊 Final Evaluation Metrics:")
    print(f"   🟢 Accuracy:  {accuracy_score(all_labels.flatten(), final_preds.flatten()):.4f}")
    print(f"   🟣 Precision: {precision_score(all_labels.flatten(), final_preds.flatten()):.4f}")
    print(f"   🔵 Recall:    {recall_score(all_labels.flatten(), final_preds.flatten()):.4f}")
    print(f"   🟡 F1-Score:  {best_f1_score_val:.4f}")
    print(f"   🟠 IoU:       {jaccard_score(all_labels.flatten(), final_preds.flatten()):.4f}")
    print("\n✅ Training complete!")

    
if __name__ == "_main_":
    train()