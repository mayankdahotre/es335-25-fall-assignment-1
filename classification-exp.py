import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
import os

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Shuffle X and y
shuffle_idx = np.random.permutation(y.size)
X, y = X[shuffle_idx], y[shuffle_idx]


def print_report(y_true, y_pred):
    acc = accuracy(y_pred, y_true)

    print("="*65)
    print("CLASSIFICATION PERFORMANCE REPORT")
    print("="*65)
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("-"*65)

    # Class-wise metrics
    classes = sorted(y_true.unique())
    header = f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}"
    print(header)
    print("-"*65)

    f1_scores, precisions, recalls = [], [], []
    for cls in classes:
        prec = precision(y_pred, y_true, cls)
        rec = recall(y_pred, y_true, cls)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        print(f"{cls:<10}{prec:<12.4f}{rec:<12.4f}{f1:<12.4f}")

    # Macro averages
    avg_prec, avg_rec, avg_f1 = np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

    print("-"*65)
    print(f"{'Macro Avg':<10}{avg_prec:<12.4f}{avg_rec:<12.4f}{avg_f1:<12.4f}")
    print("="*65)


def prepare_dataset():
    total_size = len(X)
    train_size = int(total_size * 0.7)

    X_train = pd.DataFrame(X[:train_size])
    y_train = pd.Series(y[:train_size])
    X_test = pd.DataFrame(X[train_size:])
    y_test = pd.Series(y[train_size:])

    return X_train, y_train, X_test, y_test


def create_depth_analysis_plot(tested_depths, fold_depths):
    """Create visualization of depth selection across folds"""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    from collections import Counter
    depth_counts = Counter(fold_depths)

    plt.figure(figsize=(10, 6))

    # Subplot 1: Depth frequency
    plt.subplot(1, 2, 1)
    depths_list = list(depth_counts.keys())
    counts_list = list(depth_counts.values())
    bars = plt.bar(depths_list, counts_list, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Optimal Depth')
    plt.ylabel('Frequency (Number of Folds)')
    plt.title('Optimal Depth Distribution Across Folds')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, counts_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(count), ha='center', va='bottom', fontweight='bold')

    # Subplot 2: Fold-wise depth selection
    plt.subplot(1, 2, 2)
    fold_numbers = list(range(len(fold_depths)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(set(fold_depths))))
    color_map = {depth: colors[i] for i, depth in enumerate(sorted(set(fold_depths)))}
    bar_colors = [color_map[depth] for depth in fold_depths]

    bars = plt.bar(fold_numbers, fold_depths, color=bar_colors, edgecolor='black', alpha=0.8)
    plt.xlabel('Fold Number')
    plt.ylabel('Optimal Depth')
    plt.title('Optimal Depth per Fold')
    plt.xticks(fold_numbers)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, depth in zip(bars, fold_depths):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(depth), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save individual plots separately
    plt.figure(figsize=(6, 5))
    plt.bar(depths_list, counts_list, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Optimal Depth')
    plt.ylabel('Frequency (Number of Folds)')
    plt.title('Optimal Depth Distribution Across Folds')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (depth, count) in enumerate(zip(depths_list, counts_list)):
        plt.text(depth, count + 0.05, str(count), ha='center', va='bottom', fontweight='bold')

    plt.savefig("Asst0_plots/depth_frequency_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save fold-wise depth plot separately
    plt.figure(figsize=(6, 5))
    fold_numbers = list(range(len(fold_depths)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(set(fold_depths))))
    color_map = {depth: colors[i] for i, depth in enumerate(sorted(set(fold_depths)))}
    bar_colors = [color_map[depth] for depth in fold_depths]

    plt.bar(fold_numbers, fold_depths, color=bar_colors, edgecolor='black', alpha=0.8)
    plt.xlabel('Fold Number')
    plt.ylabel('Optimal Depth')
    plt.title('Optimal Depth per Fold')
    plt.xticks(fold_numbers)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, depth in enumerate(fold_depths):
        plt.text(i, depth + 0.05, str(depth), ha='center', va='bottom', fontweight='bold')

    plt.savefig("Asst0_plots/depth_per_fold.png", dpi=300, bbox_inches='tight')
    plt.close()

# For plotting
scatter = plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Dataset")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.gca().add_artist(plt.legend(*scatter.legend_elements(), loc="upper right", title="Labels"))
os.makedirs("Asst0_plots", exist_ok=True)
plt.savefig("Asst0_plots/dataset.png")


# Write the code for Q2 a) and b) below. Show your results.
# Q2a
def evaluate_tree():
    X_train, y_train, X_test, y_test = prepare_dataset()

    # Test different criteria
    criteria = ['information_gain', 'gini_index']
    results = {}

    for criterion in criteria:
        print(f"\n Testing with {criterion.upper()} criterion:")
        tree = DecisionTree(criterion=criterion, max_depth=5)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)

        acc = accuracy(y_pred, y_test)
        results[criterion] = acc
        print_report(y_test, y_pred)

        # Print tree depth info
        print(f"Tree depth: {tree.root.get_depth()}")
        print(f"Number of leaf nodes: {tree.root.count_leaves()}")

    # Compare criteria
    print("\n" + "="*60)
    print("CRITERION COMPARISON")
    print("="*60)
    for criterion, acc in results.items():
        print(f"{criterion.upper():<20}: {acc:.4f} ({acc*100:.2f}%)")

    best_criterion = max(results, key=results.get)
    print(f"\n Best performing criterion: {best_criterion.upper()}")
    print("="*60)


# Q2b
# Use 5 fold cross-validation on the dataset. Using nested cross-validation find the optimum depth of the tree. [1 mark] Implement cross-validation from scratch.
def inner_cross_validation(X, y, n_folds, depths):
    fold_size = len(X) // n_folds
    d_acc = {}
    for depth in depths:
        acc = []
        for i in range(n_folds):
            X_train = pd.concat([X[:i * fold_size], X[(i + 1) * fold_size:]])
            y_train = pd.concat([y[:i * fold_size], y[(i + 1) * fold_size:]])
            X_val = X[i * fold_size:(i + 1) * fold_size]
            y_val = y[i * fold_size:(i + 1) * fold_size]
            
            tree = DecisionTree(criterion='information_gain', max_depth=depth)
            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_val)
            acc.append(accuracy(y_pred.to_numpy(), y_val.to_numpy()))

        d_acc[depth] = np.mean(acc).item()
    d_acc = dict(sorted(d_acc.items(), key=lambda item: (item[1], item[0]), reverse=True))
    return d_acc

def outer_cross_validation(X, y, n_folds, depths):
    fold_size = len(X) // n_folds
    acc = []
    depths_per_fold = []
    
    for i in range(n_folds):
        X_train = pd.concat([X[:i * fold_size], X[(i + 1) * fold_size:]])
        y_train = pd.concat([y[:i * fold_size], y[(i + 1) * fold_size:]])
        X_test = X[i * fold_size:(i + 1) * fold_size]
        y_test = y[i * fold_size:(i + 1) * fold_size]
        
        depths_acc = inner_cross_validation(X_train, y_train, 5, depths)
        best_depth = list(depths_acc.keys())[0]
        tree = DecisionTree(criterion='information_gain', max_depth=best_depth)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        acc.append(accuracy(y_pred.to_numpy(), y_test.to_numpy()))
        depths_per_fold.append(best_depth)
        print(f"Fold {i}, Mean Depth Accuracies on Validation {depths_acc}")
        print (f"Fold {i}, Accuracy on Test dataset: {acc[-1]}")
        
    return np.mean(acc), depths_per_fold

def evaluate_k_fold_nested_cross_validation(X, y, n_folds, depths):
    acc, fold_depths = outer_cross_validation(X, y, n_folds, depths)

    # Depth analysis
    from collections import Counter
    depth_counts = Counter(fold_depths)
    print(f"\nDepth frequency analysis:")
    for depth, count in sorted(depth_counts.items()):
        percentage = (count / len(fold_depths)) * 100
        print(f"  Depth {depth}: {count}/{len(fold_depths)} folds ({percentage:.1f}%)")

    most_common_depth = depth_counts.most_common(1)[0][0]
    print(f"\nRecommended optimal depth: {most_common_depth}")
    print("="*60)

    # Create depth performance visualization
    create_depth_analysis_plot(depths, fold_depths)

    print(f"Mean Accuracy accross {n_folds} folds: ", acc)
    print("Depths: ", depths)

    return acc, fold_depths

def main():
    print("Dataset size:", len(X), "samples with", X.shape[1], "features")

    # Run single tree evaluation
    evaluate_tree()

    # Run cross-validation
    depths_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cv_acc, cv_depths = evaluate_k_fold_nested_cross_validation(pd.DataFrame(X), pd.Series(y), 5, depths_to_test)

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Cross-validation: 5-fold nested CV with depths {depths_to_test}")
    print(f"Best CV accuracy: {cv_acc:.4f} ({cv_acc*100:.2f}%)")
    print(f"Recommended depth: {max(set(cv_depths), key=cv_depths.count)}")
    print("="*80)

if __name__ == "__main__":
    main()
