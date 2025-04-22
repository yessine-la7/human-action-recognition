import os
import torch
from data_loader_batch import get_data_loaders
from create_model import create_custom_model
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

# Function to load the trained model
def load_model(model_path, num_classes, device):
    model = create_custom_model(num_classes)

    # Map model to the correct device (CPU or GPU)
    if device.type == 'cpu':
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model.load_state_dict(torch.load(model_path))

    return model.to(device).eval()

# Function to denormalize images
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Denormalize the image
    return tensor

# Function to make predictions on the test set
def predict():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load test data
    _, _, test_loader, label_encoder = get_data_loaders(
        os.path.join("..", "Human Action Recognition", "Training_set.csv"),
        os.path.join("..", "Human Action Recognition", "Testing_set.csv"),
    )

    print(f"\nNumber of test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(label_encoder.classes_)}\n")

    # Load the best model
    model = load_model("best_model.pth", len(label_encoder.classes_), device)
    print("Model loaded successfully.")

    # Run inference on the test data
    all_preds, all_probs, all_inputs = [], [], []
    results_count = 5

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)  # Move inputs to GPU
            outputs = model(inputs)  # Get model outputs

            # Get predictions and probabilities
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_inputs.append(inputs.cpu())  # Store inputs for visualization

            probabilities = F.softmax(outputs, dim=1)
            max_probs = probabilities.max(1)[0]
            all_probs.extend(max_probs.cpu().numpy())

    # Combine input tensors for easier random sampling
    all_inputs = torch.cat(all_inputs, dim=0)

    # Normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Randomly select indices for display
    random_indices = random.sample(range(len(all_inputs)), results_count)

    # Display the selected predictions
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(random_indices):
        plt.subplot(1, results_count, i + 1)

        # Denormalize and display the image
        img = denormalize(all_inputs[idx], mean, std).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)  # Clip values to [0, 1]
        plt.imshow(img)

        # Show prediction and probability
        pred_class = label_encoder.inverse_transform([all_preds[idx]])[0]
        prob = all_probs[idx] * 100
        plt.title(f"Pred: {pred_class}\nProb: {prob:.2f}%")
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    predict()
