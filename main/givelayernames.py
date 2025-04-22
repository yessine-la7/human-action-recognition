import torch
import torch.nn as nn
from create_model import create_custom_model

# Dummy label encoder and device setup (for demonstration)
class DummyLabelEncoder:
    classes_ = list(range(15))  # Assuming 15 classes

label_encoder = DummyLabelEncoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the custom model
model = create_custom_model(len(label_encoder.classes_)).to(device)

# Print model architecture
print(f"\nModel architecture: \n{model}")

# Print all layer names
print("\nLayers in the model:\n")
for name, layer in model.named_modules():
    print(name)
