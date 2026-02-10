"""Training script for GCN baseline on GNNBenchmarkDataset (MNIST, CIFAR10).
Processes individual graphs (no sets) - standard graph classification."""
import torch
import torch.nn.functional as F
import numpy as np
import random
import sys
import os
from collections import defaultdict
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GNNBenchmarkDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_set_transformer.models.model_dropout import GCNGraphClassifier

import matplotlib.pyplot as plt
import pandas as pd


def load_gnn_benchmark_dataset(dataset_name):
    """Load GNNBenchmarkDataset with official splits"""
    train_dataset = GNNBenchmarkDataset(root='./data', name=dataset_name, split='train')
    val_dataset = GNNBenchmarkDataset(root='./data', name=dataset_name, split='val')
    test_dataset = GNNBenchmarkDataset(root='./data', name=dataset_name, split='test')
    
    # Ensure float features
    for dataset in [train_dataset, val_dataset, test_dataset]:
        for data in dataset:
            data.x = data.x.float()
    
    return train_dataset, val_dataset, test_dataset


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch on individual graphs."""
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        pred = model(data)  # No set_batch needed
        loss = F.cross_entropy(pred, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, num_classes):
    """Evaluate on individual graphs."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)  # No set_batch needed
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.hstack(all_labels)
    
    if num_classes == 2:
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
        precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
        auprc = auc(recall, precision)
    else:
        # Multi-class: macro-averaged metrics
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        auprc = average_precision_score(all_labels, all_probs, average='macro')
    
    return auroc, auprc


def main():
    
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'CIFAR10'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuration
    num_epochs = 200
    hidden_dim = 64
    batch_size = 64  
    learning_rate = 1e-3
    dropout = 0.3
    
    seeds = [10, 20, 30, 40, 50]
    
    print(f"\n{'#'*70}")
    print(f"{dataset_name} GCN Baseline Training")
    print(f"Official GNNBenchmarkDataset Splits")
    print(f"Processing individual graphs (no sets)")
    print(f"Seeds: {seeds}")
    print(f"{'#'*70}\n")
    
    # Load official splits
    print("Loading GNNBenchmarkDataset...")
    train_dataset, val_dataset, test_dataset = load_gnn_benchmark_dataset(dataset_name)
    print(f"Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Get dataset properties
    in_channels = train_dataset[0].x.shape[1]
    num_classes = train_dataset.num_classes
    print(f"Input channels: {in_channels}, Num classes: {num_classes}\n")
    
    # Base output directory
    base_output_dir = Path('../results') / dataset_name / f"{dataset_name}_GCN_baseline"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for aggregated results
    all_seed_results = defaultdict(list)
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"Experiment {seed_idx + 1}/{len(seeds)}")
        print(f"Seed: {seed}")
        print(f"{'='*70}\n")
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
        # Output directory
        output_dir = base_output_dir / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard PyG DataLoaders (no set processing)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        print(f"Training GCN (Seed {seed})")
        model = GCNGraphClassifier(in_channels, hidden_dim, num_classes, dropout=dropout)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        
        # Training tracking
        train_losses = []
        val_aurocs = []
        val_auprcs = []
        best_val_auroc = 0
        best_val_auprc = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            val_auroc, val_auprc = evaluate(model, val_loader, device, num_classes)
            
            train_losses.append(train_loss)
            val_aurocs.append(val_auroc)
            val_auprcs.append(val_auprc)
            
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_auprc = val_auprc
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}")
        
        print(f"Best Val AUROC: {best_val_auroc:.4f}, AUPRC: {best_val_auprc:.4f}")
        
        # Evaluate on test set
        model.load_state_dict(best_model_state)
        test_auroc, test_auprc = evaluate(model, test_loader, device, num_classes)
        print(f"Test AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")
        
        # Store for aggregation
        all_seed_results['val_auroc'].append(best_val_auroc)
        all_seed_results['val_auprc'].append(best_val_auprc)
        all_seed_results['test_auroc'].append(test_auroc)
        all_seed_results['test_auprc'].append(test_auprc)
        
        # Save plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(train_losses, label='GCN')
        axes[0].set_title('Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        axes[1].plot(val_aurocs, label='GCN')
        axes[1].set_title('Validation AUROC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUROC')
        axes[1].legend()
        
        axes[2].plot(val_auprcs, label='GCN')
        axes[2].set_title('Validation AUPRC')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUPRC')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=150)
        plt.close()
        
        # Save CSV results
        summary = pd.DataFrame({
            'Model': ['GCN'],
            'Best Val AUROC': [best_val_auroc],
            'Best Val AUPRC': [best_val_auprc],
            'Test AUROC': [test_auroc],
            'Test AUPRC': [test_auprc],
        })
        summary.to_csv(output_dir / 'results_summary.csv', index=False)
        
        # Save training history
        history = pd.DataFrame({
            'epoch': list(range(1, num_epochs + 1)),
            'train_loss': train_losses,
            'val_auroc': val_aurocs,
            'val_auprc': val_auprcs,
        })
        history.to_csv(output_dir / 'training_history.csv', index=False)
        
        print(f"\nSaved results to {output_dir}")
    
    # Aggregate results
    print(f"\n{'='*70}")
    print(f"AGGREGATED RESULTS FOR GCN BASELINE")
    print(f"{'='*70}\n")
    
    aggregated_summary = pd.DataFrame({
        'Model': ['GCN'],
        'Val_AUROC_Mean': [np.mean(all_seed_results['val_auroc'])],
        'Val_AUROC_Std': [np.std(all_seed_results['val_auroc'])],
        'Val_AUPRC_Mean': [np.mean(all_seed_results['val_auprc'])],
        'Val_AUPRC_Std': [np.std(all_seed_results['val_auprc'])],
        'Test_AUROC_Mean': [np.mean(all_seed_results['test_auroc'])],
        'Test_AUROC_Std': [np.std(all_seed_results['test_auroc'])],
        'Test_AUPRC_Mean': [np.mean(all_seed_results['test_auprc'])],
        'Test_AUPRC_Std': [np.std(all_seed_results['test_auprc'])],
    })
    
    aggregated_summary.to_csv(base_output_dir / 'aggregated_results.csv', index=False)
    print(aggregated_summary)
    print(f"\nSaved to {base_output_dir / 'aggregated_results.csv'}")
    
    print(f"\n{'#'*70}")
    print(f"ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: ../results/{dataset_name}/{dataset_name}_GCN_baseline/")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
