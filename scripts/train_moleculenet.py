from time import perf_counter
import torch
import torch_geometric
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.datasets import MoleculeNet
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from pathlib import Path
from datetime import datetime
from graph_set_transformer.models import (
    SetTransformerGraphClassifier,
    DeepSetGraphClassifier,
    GraphSetTransformerClassifier,
)

from graph_set_transformer.utils import molecule_net_loader
from graph_set_transformer.utils import tdc_adme_loader

from graph_set_transformer.data import (
    SetDataset,
    BalancedSetBatchSampler,
    collate_sets,
    make_label_homogeneous_sets,
    make_label_homogeneous_sets_rand_card,
)

import matplotlib.pyplot as plt
import pandas as pd


def auprc_score(y_true, y_pred):
    precision, recall, threshold = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


def get_model(model_name, in_channels, hidden_dim, num_classes):
    if model_name == "SetTransformer":
        return SetTransformerGraphClassifier(in_channels, hidden_dim, num_classes)
    elif model_name == "DeepSets":
        return DeepSetGraphClassifier(in_channels, hidden_dim, num_classes)
    elif model_name == "GraphSetConv":
        return GraphSetTransformerClassifier(in_channels, hidden_dim, num_classes)


def load_dataset(dataset_name):
    """Load datasets"""

    def transform(data):
        data.x = data.x.float()
        return data

    dataset = MoleculeNet(root="./data", name=dataset_name, pre_transform=transform)
    return dataset


def calculate_class_weights(dataset, num_classes=2):
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for data in dataset:
        class_counts[data.y.item()] += 1

    # Calculate weights as inverse frequency
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)

    # Normalize weights (optional, but helps with numerical stability)
    class_weights = class_weights / class_weights.sum() * num_classes

    print(f"\nClass distribution in training set:")
    for i in range(num_classes):
        print(
            f"  Class {i}: {int(class_counts[i])} samples ({class_counts[i]/total_samples*100:.2f}%)"
        )
    print(f"Class weights: {class_weights.tolist()}")

    return class_weights


def train_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    for data, set_batch, targets in loader:
        data = data.to(device)
        set_batch = set_batch.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        pred = model(data, set_batch)

        # Use weighted cross entropy if class weights are provided
        if class_weights is not None:
            loss = F.cross_entropy(pred, targets, weight=class_weights)
        else:
            loss = F.cross_entropy(pred, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for data, set_batch, targets in loader:
            data = data.to(device)
            set_batch = set_batch.to(device)
            logits = model(data, set_batch)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return roc_auc_score(all_targets, all_probs)
    # return auprc_score(all_targets, all_probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_names = [
        # "CYP3A4_Substrate_CarbonMangels",
        "bace",
        "CYP2D6_Substrate_CarbonMangels",
        "CYP2C9_Substrate_CarbonMangels",
        "bbbp",
        "BBB_Martins",
        "Pgp_Broccatelli",
    ]

    # model_names = ["GraphSetConv", "SetTransformer", "DeepSets"]
    model_names = ["SetTransformer"]
    # model_names = ["GraphSetConv"]

    # settransformer works better with 5e-4 for moleculenet and better with 1e-4 for tdc
    learning_rates = {
        "SetTransformer": 1e-3,
        "DeepSets": 1e-3,
        "GraphSetConv": 1e-4,
    }
    hidden_dims = {"SetTransformer": 64, "DeepSets": 64, "GraphSetConv": 64}
    set_sizes = [10]
    # set_sizes = [5, 10, 20]
    # set_sizes = [20, 10, 5]
    num_epochs = 1000
    batch_size = 32
    num_trials = 5
    use_class_weights = False

    all_results = {
        set_size: {
            model_name: {
                "train_loss_per_trial": [],
                "val_auroc_per_trial": [],
                "test_auroc_per_trial": [],
            }
            for model_name in model_names
        }
        for set_size in set_sizes
    }

    for dataset_name in dataset_names:
        print(f"\n{'*'*80}")
        print(f"* PROCESSING DATASET: {dataset_name}")
        print(f"{'*'*80}\n")

        if dataset_name in ["bace", "bbbp"]:
            train_dataset, val_dataset, test_dataset, tasks = molecule_net_loader(
                dataset_name,
                f"data/moleculenet/{dataset_name}.csv.xz",
                use_scaffold_split=True,
            )
        else:
            train_dataset, val_dataset, test_dataset, tasks = tdc_adme_loader(
                dataset_name
            )

        in_channels = train_dataset[0].x.shape[1]
        num_classes = 2

        # Calculate class weights from the training dataset
        if use_class_weights:
            class_weights = calculate_class_weights(train_dataset, num_classes)
            class_weights = class_weights.to(device)
            print(f"\nUsing class weights: {class_weights.cpu().tolist()}\n")
        else:
            class_weights = None
            print("\nNot using class weights\n")

        # Iterate over set sizes
        for set_size in set_sizes:
            if len(train_dataset) < 500 and set_size > 5:
                continue

            print(f"\n{'*'*70}")
            print(f"* SET SIZE: {set_size}")
            print(f"{'*'*70}")

            # Create sets of graphs with homogeneous labels
            val_sets = make_label_homogeneous_sets(val_dataset, set_size)
            test_sets = make_label_homogeneous_sets(test_dataset, set_size)

            # val_sets = make_label_homogeneous_sets_rand_card(
            #     val_dataset, card_max=set_size, shuffle=False
            # )
            # test_sets = make_label_homogeneous_sets_rand_card(
            #     test_dataset, card_max=set_size, shuffle=False
            # )

            # Create SetDatasets
            val_set_dataset = SetDataset(val_sets)
            test_set_dataset = SetDataset(test_sets)

            print(
                f"N Val sets: {len(val_set_dataset)},N Test sets: {len(test_set_dataset)}"
            )

            # Create DataLoaders
            val_loader = TorchDataLoader(
                val_set_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_sets,
            )
            test_loader = TorchDataLoader(
                test_set_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_sets,
            )

            for trial in range(num_trials):
                print(f"\n{'#'*60}")
                print(f"# SET SIZE: {set_size} - TRIAL {trial + 1}/{num_trials}")
                print(f"{'#'*60}")
                # Results for this trial (for plotting the first trial)
                trial_results = {
                    model_name: {"train_loss": [], "val_auroc": []}
                    for model_name in model_names
                }

                # train_sets = make_label_homogeneous_sets(
                #     train_dataset, set_size, shuffle=False
                # )

                train_sets = make_label_homogeneous_sets_rand_card(
                    train_dataset, card_max=set_size, shuffle=False
                )
                train_set_dataset = SetDataset(train_sets)
                balanced_sampler = BalancedSetBatchSampler(
                    train_set_dataset, batch_size
                )

                train_loader = TorchDataLoader(
                    train_set_dataset,
                    # batch_size=batch_size,
                    # shuffle=True,
                    batch_sampler=balanced_sampler,
                    collate_fn=collate_sets,
                )
                end = perf_counter()

                # Train each model
                for model_name in model_names:
                    print(f"\n{'='*50}")
                    print(
                        f"Set Size {set_size} - Trial {trial + 1} - Training {model_name}"
                    )
                    print(f"{'='*50}")

                    model = get_model(
                        model_name, in_channels, hidden_dims[model_name], num_classes
                    )
                    model = model.to(device)

                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=learning_rates[model_name],
                        weight_decay=0.01,
                        betas=(0.9, 0.999),
                    )

                    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    #     optimizer, T_max=num_epochs, eta_min=1e-5
                    # )

                    best_val_auroc = 0
                    best_model_state = None

                    for epoch in range(num_epochs):
                        start = perf_counter()
                        # print(f"time 1: {end - start}")

                        # if epoch == 0 and trial == 0:
                        #     print(f"\nVerifying batch balance for epoch 1:")
                        #     for i, (data, set_batch, targets) in enumerate(
                        #         train_loader
                        #     ):
                        #         class_counts = torch.bincount(targets)
                        #         print(f"  Batch {i}: {class_counts.tolist()}")
                        #         # if i >= 2:  # Just show first 3 batches
                        #         #     break

                        start = perf_counter()
                        # Train with class weights
                        train_loss = train_epoch(
                            model, train_loader, optimizer, device, class_weights
                        )

                        end = perf_counter()
                        # print(f"time 2: {end - start}")

                        # scheduler.step()

                        # Validate
                        start = perf_counter()
                        val_auroc = evaluate(model, val_loader, device)

                        end = perf_counter()
                        # print(f"time 3: {end - start}")

                        # Save Results
                        trial_results[model_name]["train_loss"].append(train_loss)
                        trial_results[model_name]["val_auroc"].append(val_auroc)

                        # Add warmup for imbalanced data sets and deepsets
                        if val_auroc > best_val_auroc and epoch > 150:
                            best_val_auroc = val_auroc
                            best_model_state = {
                                k: v.cpu().clone()
                                for k, v in model.state_dict().items()
                            }

                        if (epoch + 1) % 10 == 0:
                            print(
                                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val AUROC: {val_auroc:.4f}"
                            )

                    print(f"Best Val AUROC for {model_name}: {best_val_auroc:.4f}")

                    # Evaluate on test set
                    model.load_state_dict(best_model_state)
                    test_auroc = evaluate(model, test_loader, device)
                    print(f"Test AUROC for {model_name}: {test_auroc:.4f}")

                    print(
                        f"Best Val AUROC for {model_name} (Set Size {set_size}, Trial {trial + 1}): {best_val_auroc:.4f}"
                    )
                    print(
                        f"Test AUROC for {model_name} (Set Size {set_size}, Trial {trial + 1}): {test_auroc:.4f}"
                    )

                    # Store results across trials
                    all_results[set_size][model_name]["train_loss_per_trial"].append(
                        trial_results[model_name]["train_loss"]
                    )
                    all_results[set_size][model_name]["val_auroc_per_trial"].append(
                        trial_results[model_name]["val_auroc"]
                    )
                    all_results[set_size][model_name]["test_auroc_per_trial"].append(
                        test_auroc
                    )

        print(f"\n{'='*70}")
        print("FINAL RESULTS ACROSS ALL SET SIZES, MODELS, AND TRIALS")
        print(f"{'='*70}\n")

        summary_data = []
        for set_size in set_sizes:
            print(f"\nSET SIZE: {set_size}")
            print(f"{'-'*60}")
            for model_name in model_names:
                test_aurocs = all_results[set_size][model_name]["test_auroc_per_trial"]
                mean_test_auroc = np.mean(test_aurocs)
                std_test_auroc = np.std(test_aurocs)

                print(f"{model_name}:")
                print(f"  Test AUROC across {num_trials} trials: {test_aurocs}")
                print(f"  Mean ± Std: {mean_test_auroc:.4f} ± {std_test_auroc:.4f}\n")

                summary_data.append(
                    {
                        "Set Size": set_size,
                        "Model": model_name,
                        "Mean Test AUROC": mean_test_auroc,
                        "Std Test AUROC": std_test_auroc,
                        "All Test AUROCs": test_aurocs,
                    }
                )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weight_suffix = "_weighted" if use_class_weights else ""

        plot_filename = f"model_comparison_{dataset_name}{weight_suffix}.png"
        csv_filename = f"model_comparison_{dataset_name}{weight_suffix}.csv"

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_filename, index=False)
        print(f"Saved summary to {csv_filename}")

        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(summary_df[["Set Size", "Model", "Mean Test AUROC", "Std Test AUROC"]])
        print("\n")


if __name__ == "__main__":
    main()
