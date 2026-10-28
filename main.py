"""
Project 07: Multi-Layer Perceptron (MLP) with Dropout Regularization
Dataset: Zalando FashionMNIST (10 apparel classes, 28x28 grayscale)
Framework: PyTorch

Key Features:
- Multi-layer perceptron with multiple hidden layers (784 -> 512 -> 256 -> 128 -> 10)
- Inverted Dropout regularization (Srivastava et al., 2014)
- Controlled ablation study: Baseline MLP (p=0.0) vs Regularized MLP (p=0.3)
- Explicit demonstration of model.train() vs model.eval() modes
- Comprehensive 4-panel diagnostic visual artifact:
  1) Cross-Entropy Loss convergence (Train vs Test)
  2) Generalization Accuracy Gap analysis
  3) Empirical Neuron Activation & Dropout Mask visualization
  4) Normalized 10-Class Confusion Matrix for Regularized MLP
"""

import os
import sys
import types
import random
import numpy as np

# Workaround for Windows environments where WDAC blocks unsigned bz2/lzma DLLs
if 'bz2' not in sys.modules:
    sys.modules['bz2'] = types.SimpleNamespace(open=open)
if 'lzma' not in sys.modules:
    sys.modules['lzma'] = types.SimpleNamespace(open=open)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FashionMLP(nn.Module):
    """
    Deep Multi-Layer Perceptron (MLP) with configurable Dropout regularization.
    
    Architecture:
      Input (784) -> Linear(512) -> ReLU -> [Dropout(p)] 
                  -> Linear(256) -> ReLU -> [Dropout(p)] 
                  -> Linear(128) -> ReLU -> [Dropout(p)] 
                  -> Linear(10)  -> Softmax Logits
    """
    def __init__(self, input_dim=784, hidden_dims=(512, 256, 128), num_classes=10, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten spatial dimensions: [B, 1, 28, 28] -> [B, 784]
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_data_loaders(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader, train_dataset.classes


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()  # Dropout active
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()  # Dropout inactive (weights scaled automatically by inverted dropout)
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_targets)


def run_experiment(model, train_loader, test_loader, epochs, lr, device, experiment_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    print(f"\nTraining [{experiment_name}] across {epochs} Epochs...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc, preds, targets = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)
        
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.2f}%")
        
    return history, preds, targets


def compute_confusion_matrix(preds, targets, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


def generate_visual_artifact(hist_no_dropout, hist_dropout, cm, class_names, sample_img, model_dropout, device, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    epochs = range(1, len(hist_dropout['train_loss']) + 1)
    
    # -------------------------------------------------------------
    # Panel 1: Loss Comparison (Overfitting Dynamics)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(epochs, hist_no_dropout['train_loss'], 'b--', label='No Dropout - Train Loss', alpha=0.7)
    ax1.plot(epochs, hist_no_dropout['test_loss'], 'r--', label='No Dropout - Test Loss (Overfitting)', linewidth=2.0)
    ax1.plot(epochs, hist_dropout['train_loss'], 'b-', label='With Dropout (p=0.3) - Train Loss', linewidth=2.0)
    ax1.plot(epochs, hist_dropout['test_loss'], 'g-', label='With Dropout (p=0.3) - Test Loss (Regularized)', linewidth=2.5)
    ax1.set_title("Cross-Entropy Loss vs. Epochs\nOverfitting Mitigation via Dropout", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    
    # -------------------------------------------------------------
    # Panel 2: Generalization Gap (Train vs. Test Accuracy)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    gap_no_dropout = np.array(hist_no_dropout['train_acc']) - np.array(hist_no_dropout['test_acc'])
    gap_dropout = np.array(hist_dropout['train_acc']) - np.array(hist_dropout['test_acc'])
    
    ax2.plot(epochs, hist_no_dropout['test_acc'], 'm--', label='No Dropout - Test Acc', linewidth=1.8)
    ax2.plot(epochs, hist_dropout['test_acc'], 'darkgreen', label='With Dropout - Test Acc', linewidth=2.2)
    ax2.plot(epochs, gap_no_dropout, 'red', linestyle='-.', label=f'No Dropout Gap (Delta={gap_no_dropout[-1]:.1f}%)', alpha=0.8)
    ax2.plot(epochs, gap_dropout, 'blue', linestyle='-.', label=f'With Dropout Gap (Delta={gap_dropout[-1]:.1f}%)', alpha=0.8)
    ax2.set_title("Generalization Accuracy & Overfitting Gap\n(Train Accuracy - Test Accuracy)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy / Gap (%)", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='center right', frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Empirical Dropout Mask & Hidden Unit Activations
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    # Pass a sample batch through the first linear layer to visualize activations
    sample_img = sample_img.to(device)
    flat = sample_img.view(sample_img.size(0), -1)
    first_linear = model_dropout.classifier[0]  # Linear(784, 512)
    h_act = torch.relu(first_linear(flat))[:16, :32].cpu().detach().numpy()
    
    # Simulate a 0.3 dropout mask
    mask = (np.random.rand(*h_act.shape) >= 0.3).astype(float)
    h_dropped = (h_act * mask) / (1.0 - 0.3)
    
    im3 = ax3.imshow(h_dropped, aspect='auto', cmap='magma')
    ax3.set_title("Hidden Layer Activations with Inverted Dropout (p=0.3)\nZeroed Neurons Prevent Feature Co-adaptation", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Neuron Index (First 32 Units)", fontsize=11)
    ax3.set_ylabel("Batch Sample Index (16 Samples)", fontsize=11)
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label("Scaled Activation Magnitude", fontsize=10)

    # -------------------------------------------------------------
    # Panel 4: Normalized Confusion Matrix for Regularized MLP
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im4 = ax4.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax4.set_title("Confusion Matrix (Regularized MLP With Dropout)\nNormalized Per-Class Accuracy", fontsize=12, fontweight='bold')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label("Accuracy Ratio", fontsize=10)
    
    tick_marks = np.arange(len(class_names))
    ax4.set_xticks(tick_marks)
    ax4.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax4.set_yticks(tick_marks)
    ax4.set_yticklabels(class_names, fontsize=9)
    ax4.set_xlabel("Predicted Label", fontsize=11)
    ax4.set_ylabel("True Ground Truth Label", fontsize=11)
    
    # Print diagonal values
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            if val >= 0.05:  # Only label visible confusions
                ax4.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.suptitle("FashionMNIST Multi-Layer Perceptron (MLP) with Dropout Regularization", fontsize=16, fontweight='heavy', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisual diagnostic report saved successfully at: {os.path.abspath(save_path)}")


def main():
    print("=" * 65)
    print("   PROJECT 07: MLP WITH DROPOUT REGULARIZATION ON FASHION-MNIST")
    print("=" * 65)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device    : {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    save_path = os.path.join(script_dir, "mlp_dropout_regularization_results.png")
    
    # Hyperparameters
    batch_size = 128
    epochs = 6
    learning_rate = 0.001
    
    # Load Data
    train_loader, test_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    print(f"Dataset             : FashionMNIST (60k Train, 10k Test)")
    print(f"Classes (10 Items)  : {', '.join(class_names)}")
    print(f"Architecture        : 784 -> 512 -> 256 -> 128 -> 10 (Deep MLP)")
    print(f"Ablation Experiment : Baseline (No Dropout p=0.0) vs Regularized (Dropout p=0.3)")
    print("-" * 65)
    
    # Model 1: Baseline MLP (No Dropout)
    set_seed(42)
    model_no_dropout = FashionMLP(dropout_p=0.0).to(device)
    hist_no_dropout, _, _ = run_experiment(
        model_no_dropout, train_loader, test_loader, epochs=epochs, lr=learning_rate,
        device=device, experiment_name="Baseline MLP (No Dropout p=0.0)"
    )
    
    # Model 2: Regularized MLP (With Dropout p=0.3)
    set_seed(42)
    model_dropout = FashionMLP(dropout_p=0.3).to(device)
    hist_dropout, test_preds, test_targets = run_experiment(
        model_dropout, train_loader, test_loader, epochs=epochs, lr=learning_rate,
        device=device, experiment_name="Regularized MLP (Dropout p=0.3)"
    )
    
    # Compute Final Metrics
    final_no_drop_acc = hist_no_dropout['test_acc'][-1]
    final_drop_acc = hist_dropout['test_acc'][-1]
    gap_no_drop = hist_no_dropout['train_acc'][-1] - final_no_drop_acc
    gap_drop = hist_dropout['train_acc'][-1] - final_drop_acc
    
    print("\n" + "=" * 65)
    print("                    EXPERIMENTAL COMPARISON")
    print("=" * 65)
    print(f" Model Variant                 | Train Acc | Test Acc  | Overfitting Gap (Delta)")
    print("-" * 65)
    print(f" Baseline MLP (No Dropout)     |  {hist_no_dropout['train_acc'][-1]:.2f}%   |  {final_no_drop_acc:.2f}%   |      +{gap_no_drop:.2f}%")
    print(f" Regularized MLP (Dropout 0.3) |  {hist_dropout['train_acc'][-1]:.2f}%   |  {final_drop_acc:.2f}%   |      +{gap_drop:.2f}%")
    print("-" * 65)
    print(f" Overfitting Gap Reduction     :  {gap_no_drop - gap_drop:.2f}% smaller gap with Dropout!")
    print("=" * 65)
    
    # Generate Visual Artifact
    sample_images, _ = next(iter(test_loader))
    cm = compute_confusion_matrix(test_preds, test_targets, num_classes=10)
    generate_visual_artifact(
        hist_no_dropout, hist_dropout, cm, class_names,
        sample_images, model_dropout, device, save_path
    )


if __name__ == "__main__":
    main()

