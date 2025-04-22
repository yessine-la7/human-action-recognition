import os
import torch
import torch.nn as nn
import torch.optim as optim
from create_model import create_custom_model
from data_loader_batch import get_data_loaders
import matplotlib.pyplot as plt
import time
from hooks import register_hooks, save_activations
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import numpy as np

# Dictionary to store activations in case you want to output them here.
activations = {}


def train():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load data
    train_loader, val_loader, _, label_encoder = get_data_loaders(
        os.path.join("..", "Human Action Recognition", "Training_set.csv"),
        os.path.join("..", "Human Action Recognition", "Testing_set.csv"),
    )

    # Create model
    model = create_custom_model(len(label_encoder.classes_)).to(device)

    # Register hooks on the specific custom layers
    register_hooks(model, activations)

    # Create a directory to save activations
    if not os.path.exists("activations"):
        os.makedirs("activations")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    print("\nStarting training ...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_train_labels = []
        all_train_preds = []

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(predicted.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:  # Print every 50 batches
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Acc: {100.0 * correct / total:.2f}%"
                )

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Calculate training metrics
        train_accuracy = correct / total
        train_precision = precision_score(
            all_train_labels, all_train_preds, average="weighted"
        )
        train_recall = recall_score(
            all_train_labels, all_train_preds, average="weighted"
        )
        train_f1 = f1_score(all_train_labels, all_train_preds, average="weighted")

        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())

        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Calculate validation metrics
        val_accuracy = correct / total
        val_precision = precision_score(
            all_val_labels, all_val_preds, average="weighted"
        )
        val_recall = recall_score(all_val_labels, all_val_preds, average="weighted")
        val_f1 = f1_score(all_val_labels, all_val_preds, average="weighted")

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
            f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}"
        )

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes")

    # Calculate overall metrics
    print("\nOverall Training Metrics:")
    print(f"Loss: {np.mean(train_losses):.4f}")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"F1-score: {train_f1:.4f}")

    print("\nOverall Validation Metrics:")
    print(f"Loss: {np.mean(val_losses):.4f}")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1-score: {val_f1:.4f}")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curves.png")
    plt.show()

    # Plot confusion matrix
    cm = confusion_matrix(all_val_labels, all_val_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()