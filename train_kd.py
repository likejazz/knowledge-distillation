"""End-to-end runnable knowledge-distillation demo.

- Mock teacher: larger MLP, pre-trained with plain cross-entropy.
- Mock student: smaller MLP, trained with train_knowledge_distillation below.
- Synthetic 2D, 3-class dataset (torch only, no extra deps).
- Saves a loss/accuracy chart PNG after training.

Run:
    uv sync
    uv run python train_kd.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_SAMPLES = 1500
NUM_CLASSES = 3
BATCH_SIZE = 64
TEACHER_EPOCHS = 30
DISTILL_EPOCHS = 20
T = 3.0
ALPHA = 0.1
LR = 1e-2
CHART_PATH = "kd_training_curve.png"

DEVICE = torch.device("cuda")


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA build of torch and an NVIDIA GPU are required."
        )


def train_knowledge_distillation(teacher, student, train_loader, optimizer, epoch, device, T=3.0, alpha=0.1):
    """
    Args:
        teacher: Pre-trained teacher model (in eval mode)
        student: Student model to be trained
        T: Temperature hyperparameter for softening probability distributions
        alpha: Weight assigned to the standard cross-entropy loss (hard targets)

    Returns:
        dict with averaged 'total', 'hard', 'soft' losses for the epoch
        (added so callers can log/visualize training progress).
    """
    teacher.eval()
    student.train()

    ce_loss_fn = nn.CrossEntropyLoss()
    # KL Divergence is used for the soft target loss
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    sum_total, sum_hard, sum_soft, n_batches = 0.0, 0.0, 0.0, 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()

        # 1. Forward pass with both teacher and student
        with torch.no_grad():
            teacher_logits = teacher(data)
        student_logits = student(data)

        # 2. Compute the hard target loss (standard cross-entropy)
        hard_loss = ce_loss_fn(student_logits, target)

        # 3. Compute the soft target loss (scaled by T^2 as per Hinton et al.)
        # log_softmax is applied to student, softmax to teacher
        soft_targets = F.softmax(teacher_logits / T, dim=-1)
        soft_prob = F.log_softmax(student_logits / T, dim=-1)
        soft_loss = kl_loss_fn(soft_prob, soft_targets) * (T ** 2)

        # 4. Combined loss
        loss = (alpha * hard_loss) + ((1.0 - alpha) * soft_loss)

        # 5. Backward pass
        loss.backward()
        optimizer.step()

        sum_total += loss.item()
        sum_hard += hard_loss.item()
        sum_soft += soft_loss.item()
        n_batches += 1

    return {
        "total": sum_total / max(n_batches, 1),
        "hard": sum_hard / max(n_batches, 1),
        "soft": sum_soft / max(n_batches, 1),
    }


def make_synthetic_data(n=N_SAMPLES, num_classes=NUM_CLASSES, seed=SEED):
    """3 Gaussian blobs in 2D, one per class."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])
    per_class = n // num_classes
    xs, ys = [], []
    for c in range(num_classes):
        xs.append(centers[c] + torch.randn(per_class, 2, generator=g))
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    X = torch.cat(xs).float()
    y = torch.cat(ys)
    perm = torch.randperm(len(X), generator=g)
    return X[perm], y[perm]


class TeacherNet(nn.Module):
    """Mock teacher: wider & deeper MLP."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class StudentNet(nn.Module):
    """Mock student: small single-hidden-layer MLP."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        correct += (model(data).argmax(dim=1) == target).sum().item()
        total += target.numel()
    return correct / total


def pretrain_teacher(teacher, loader, device, epochs=TEACHER_EPOCHS, lr=LR):
    teacher.train()
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            opt.zero_grad()
            ce(teacher(data), target).backward()
            opt.step()


def plot_history(history, path=CHART_PATH):
    epochs = list(range(1, len(history["total"]) + 1))
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["total"], label="total (combined)")
    ax_loss.plot(epochs, history["hard"], label="hard (CE)")
    ax_loss.plot(epochs, history["soft"], label="soft (KL x T^2)")
    ax_loss.set_xlabel("distill epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Knowledge distillation losses")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["student_acc"], label="student")
    ax_acc.axhline(history["teacher_acc"], color="gray", linestyle="--", label="teacher (frozen)")
    ax_acc.set_xlabel("distill epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy during distillation")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    require_cuda()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = DEVICE
    print(f"device: {device} ({torch.cuda.get_device_name(device)})")

    X, y = make_synthetic_data()
    train_ds, test_ds = random_split(
        TensorDataset(X, y), [1200, 300],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=True)

    teacher = TeacherNet().to(device)
    pretrain_teacher(teacher, train_loader, device)
    teacher_acc = accuracy(teacher, test_loader, device)
    print(f"teacher test acc (frozen): {teacher_acc:.4f}")

    student = StudentNet().to(device)
    print(f"student test acc (before) : {accuracy(student, test_loader, device):.4f}")
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)

    history = {"total": [], "hard": [], "soft": [], "student_acc": [],
               "teacher_acc": teacher_acc}
    for epoch in range(1, DISTILL_EPOCHS + 1):
        stats = train_knowledge_distillation(
            teacher, student, train_loader, optimizer, epoch, device,
            T=T, alpha=ALPHA,
        )
        acc = accuracy(student, test_loader, device)
        history["total"].append(stats["total"])
        history["hard"].append(stats["hard"])
        history["soft"].append(stats["soft"])
        history["student_acc"].append(acc)
        print(f"epoch {epoch:2d} | total {stats['total']:.4f} "
              f"| hard {stats['hard']:.4f} | soft {stats['soft']:.4f} "
              f"| student acc {acc:.4f}")

    plot_history(history)
    print(f"chart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()
