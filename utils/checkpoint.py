# utils/checkpoint.py
import torch
import os

def save_checkpoint(model, optimizer, scheduler, epoch, loss, best_f1, path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'best_f1': best_f1
    }
    torch.save(state, path)
    print(f"[💾] Saved checkpoint to {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_f1 = checkpoint.get('best_f1', 0.0)

    print(f"[🔁] Loaded checkpoint from {path} (Epoch {epoch}, Best F1 {best_f1:.4f})")
    return epoch + 1, loss, best_f1
