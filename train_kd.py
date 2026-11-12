"""
Knowledge Distillation

- Mock teacher: larger MLP, pre-trained with plain cross-entropy.
- Mock student: smaller MLP, trained with train_knowledge_distillation below.
- Synthetic 2D, 3-class dataset (torch only, no extra deps).
- Saves a loss/accuracy chart PNG after training.

Methods (select with --method, default kd):
- kd: Hinton, Vinyals & Dean (2015) forward KL / soft-target cross-entropy.
- on-policy: reverse KL under the student (Agarwal et al. 2024;
  Thinking Machines Lab 2025).
"""

import argparse

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_SAMPLES = 3000
NUM_CLASSES = 3
BATCH_SIZE = 64
TEACHER_EPOCHS = 30
DISTILL_EPOCHS = 20
T = 3.0
ALPHA = 0.1
TEACHER_LR = 1e-2
DISTILL_LR = 3e-4


def train_knowledge_distillation(
    teacher,
    student,
    train_loader,
    optimizer,
    device,
    T=3.0,
    alpha=0.1,
    on_policy=False,
):
    teacher.eval()
    student.train()

    sum_total, sum_hard, sum_soft, n_batches = 0.0, 0.0, 0.0, 0

    for data, target in train_loader:
        data, target = (
            data.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )
        optimizer.zero_grad()

        with torch.no_grad():
            v = teacher(data)
        z = student(data)

        log_p = F.log_softmax(v / T, dim=-1)
        log_q = F.log_softmax(z / T, dim=-1)
        # kl_div(input, target) = KL(target || input)
        if on_policy:
            C = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
        else:
            C = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)
        soft_loss = (T**2) * C

        hard_loss = F.cross_entropy(z, target)
        total_loss = (1.0 - alpha) * soft_loss + alpha * hard_loss

        total_loss.backward()
        optimizer.step()

        sum_total += total_loss.item()
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
    remainder = n % num_classes
    xs, ys = [], []
    for c in range(num_classes):
        count = per_class + (1 if c < remainder else 0)
        xs.append(centers[c] + torch.randn(count, 2, generator=g))
        ys.append(torch.full((count,), c, dtype=torch.long))
    X = torch.cat(xs).float()
    y = torch.cat(ys)
    perm = torch.randperm(len(X), generator=g)
    return X[perm], y[perm]


class TeacherNet(nn.Module):
    """Mock teacher: wider & deeper MLP."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class StudentNet(nn.Module):
    """Mock student: small single-hidden-layer MLP."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for data, target in loader:
        data, target = (
            data.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )
        correct += (model(data).argmax(dim=1) == target).sum().item()
        total += target.numel()
    return correct / total


def pretrain_teacher(teacher, loader, device, epochs=TEACHER_EPOCHS, lr=TEACHER_LR):
    teacher.train()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in loader:
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            optimizer.zero_grad()
            criterion(teacher(data), target).backward()
            optimizer.step()


def plot_synthetic_data(X, y):
    """Scatter plot of the 2D synthetic blobs, colored by class label."""
    X_cpu, y_cpu = X.detach().cpu(), y.detach().cpu()
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(int(y_cpu.max().item()) + 1):
        pts = X_cpu[y_cpu == c]
        ax.scatter(
            pts[:, 0].numpy(),
            pts[:, 1].numpy(),
            s=12,
            alpha=0.7,
            label=f"class {c} (n={len(pts)})",
        )
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(
        f"Synthetic data: X {tuple(X.shape)} {X.dtype}, y {tuple(y.shape)} {y.dtype}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig("synthetic_data.png", dpi=150)
    plt.close(fig)


def plot_history(history, method="kd"):
    loss_epochs = list(range(1, len(history["total"]) + 1))
    # student_acc[0] is epoch 0 = before distillation.
    acc_epochs = list(range(len(history["student_acc"])))
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    if method == "kd":
        soft_label = "soft (KL x T^2)"
        loss_title = "Knowledge distillation losses"
        acc_title = "Accuracy during distillation"
        outfile = "kd_training_curve.png"
    else:
        soft_label = "soft (reverse KL x T^2)"
        loss_title = "On-policy distillation losses"
        acc_title = "Accuracy during on-policy distillation"
        outfile = "on_policy_training_curve.png"

    ax_loss.plot(loss_epochs, history["total"], label="total (combined)")
    ax_loss.plot(loss_epochs, history["hard"], label="hard (CE)")
    ax_loss.plot(loss_epochs, history["soft"], label=soft_label)
    ax_loss.set_xlabel("distill epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(loss_title)
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(
        acc_epochs, history["student_acc"], marker="o", markersize=3, label="student"
    )
    ax_acc.axhline(
        history["teacher_acc"], color="gray", linestyle="--", label="teacher (frozen)"
    )
    ax_acc.set_xlabel("epoch (0 = before distillation)")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title(acc_title)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation demo")
    parser.add_argument(
        "--method",
        choices=("kd", "on-policy"),
        default="kd",
        help=(
            "kd: Hinton et al. 2015 forward KL on teacher soft targets (default). "
            "on-policy: reverse KL under the student's own distribution."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    print(f"method: {args.method}")
    print(f"device: {device} ({torch.cuda.get_device_name(device)})")

    X, y = make_synthetic_data()
    plot_synthetic_data(X, y)
    print("data chart saved to synthetic_data.png")
    n_train = int(0.8 * len(X))
    train_ds, test_ds = random_split(
        TensorDataset(X, y),
        [n_train, len(X) - n_train],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=True)

    teacher = TeacherNet().to(device)
    pretrain_teacher(teacher, train_loader, device)
    teacher_acc = accuracy(teacher, test_loader, device)
    print(f"teacher test acc (frozen): {teacher_acc:.4f}")

    student = StudentNet().to(device)
    before_acc = accuracy(student, test_loader, device)
    print(f"student test acc (before) : {before_acc:.4f}")
    optimizer = torch.optim.Adam(student.parameters(), lr=DISTILL_LR)

    history = {
        "total": [],
        "hard": [],
        "soft": [],
        "student_acc": [before_acc],
        "teacher_acc": teacher_acc,
    }
    for epoch in range(1, DISTILL_EPOCHS + 1):
        stats = train_knowledge_distillation(
            teacher,
            student,
            train_loader,
            optimizer,
            device,
            T=T,
            alpha=ALPHA,
            on_policy=args.method == "on-policy",
        )
        acc = accuracy(student, test_loader, device)
        history["total"].append(stats["total"])
        history["hard"].append(stats["hard"])
        history["soft"].append(stats["soft"])
        history["student_acc"].append(acc)
        print(
            f"epoch {epoch:2d} | total {stats['total']:.4f} "
            f"| hard {stats['hard']:.4f} | soft {stats['soft']:.4f} "
            f"| student acc {acc:.4f}"
        )

    outfile = plot_history(history, method=args.method)
    print(f"chart saved to {outfile}")


if __name__ == "__main__":
    main()
