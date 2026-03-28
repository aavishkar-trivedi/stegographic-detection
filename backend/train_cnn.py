import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import IMAGE_SIZE, MODEL_CHECKPOINT
from .models.cnn_model import CNNModel


MEAN = 0.5
STD = 0.5


class ImageLabelDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)


def build_samples(dataset_dir: Path):
    cover_dir = dataset_dir / "cover"
    stego_dir = dataset_dir / "stego"

    if not cover_dir.exists() or not stego_dir.exists():
        raise FileNotFoundError(
            f"Expected cover/stego folders inside {dataset_dir}."
        )

    cover_samples = sorted(cover_dir.glob("*.png"))
    stego_samples = sorted(stego_dir.glob("*.png"))

    if len(cover_samples) == 0 or len(stego_samples) == 0:
        raise ValueError(f"No PNG images found in {dataset_dir}/cover or stego.")

    samples = [(path, 0.0) for path in cover_samples] + [(path, 1.0) for path in stego_samples]
    return samples, cover_samples, stego_samples


def split_train_val(cover_paths, stego_paths, val_ratio=0.1, seed=42):
    rng = random.Random(seed)

    cover_paths = list(cover_paths)
    stego_paths = list(stego_paths)
    rng.shuffle(cover_paths)
    rng.shuffle(stego_paths)

    cover_val_size = int(len(cover_paths) * val_ratio)
    stego_val_size = int(len(stego_paths) * val_ratio)

    val_cover = cover_paths[:cover_val_size]
    train_cover = cover_paths[cover_val_size:]
    val_stego = stego_paths[:stego_val_size]
    train_stego = stego_paths[stego_val_size:]

    train_samples = [(path, 0.0) for path in train_cover] + [(path, 1.0) for path in train_stego]
    val_samples = [(path, 0.0) for path in val_cover] + [(path, 1.0) for path in val_stego]

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def compute_metrics(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def build_weighted_sampler(samples):
    """Calculate proper class weight for loss function"""
    label_counts = {0.0: 0, 1.0: 0}
    for _, label in samples:
        label_counts[label] += 1
    
    n_neg = label_counts[0.0]
    n_pos = label_counts[1.0]
    
    # pos_weight: weight for positive class
    # Standard formula: neg_count / pos_count
    pos_weight = max(n_neg / n_pos, 0.1) if n_pos > 0 else 1.0
    print(f"Class balance - Negatives: {n_neg}, Positives: {n_pos}, pos_weight: {pos_weight:.4f}")
    return pos_weight


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_dir = Path(__file__).resolve().parents[1]
    train_dir = (root_dir / args.train_dir).resolve()
    test_dir = (root_dir / args.test_dir).resolve()
    checkpoint_path = (root_dir / args.checkpoint).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    _, cover_paths, stego_paths = build_samples(train_dir)
    train_samples, val_samples = split_train_val(
        cover_paths,
        stego_paths,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    test_samples, _, _ = build_samples(test_dir)

    # Compute class weight for loss function
    pos_weight = torch.tensor([build_weighted_sampler(train_samples)], device=device)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[MEAN], std=[STD]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[MEAN], std=[STD]),
    ])

    train_ds = ImageLabelDataset(train_samples, transform=train_transform)
    val_ds = ImageLabelDataset(val_samples, transform=eval_transform)
    test_ds = ImageLabelDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = CNNModel().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Disable early stopping - let it train full epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    best_val_f1 = -1.0
    best_epoch = -1

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_metrics['loss']:.5f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step()

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "train_loss": train_loss,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    print(f"Best epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("Test metrics:")
    print(f"  Loss:      {test_metrics['loss']:.5f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on BOSS cover/stego dataset")
    parser.add_argument("--train-dir", default="boss_256_0.4", help="Training dataset directory")
    parser.add_argument("--test-dir", default="boss_256_0.4_test", help="Test dataset directory")
    parser.add_argument("--checkpoint", default=MODEL_CHECKPOINT, help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())