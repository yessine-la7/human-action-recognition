import torch
import os
from PIL import Image, ImageDraw, ImageFont
from create_model import create_custom_model
from hooks import register_hooks, save_activations, visualize_kernels
from data_loader_single import ActionRecognitionDataset  # Directly import the class
from torchvision import transforms
import random

# Dictionary to store activations
activations = {}

# set pretrained model name here
model_name = "best_model.pth"


def visualize_single_example(model_name):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Instantiate the dataset directly
    dataset = ActionRecognitionDataset(
        csv_file=os.path.join("..", "Human Action Recognition", "Training_set.csv"),
        root_dir=os.path.join("..", "Human Action Recognition", "Testing_set.csv"),
        transform=transform,
    )

    # Get a random index within the dataset range
    random_idx = random.randint(0, len(dataset) - 1)

    # Get the random item from the dataset (transformed image, original image, label)
    transformed_image, original_image, label = dataset[random_idx]

    # Move the transformed image and label to the device (GPU/CPU)
    transformed_image = transformed_image.unsqueeze(0).to(device)  # Add batch dimension
    label = torch.tensor([label]).to(
        device
    )  # Convert label to tensor and move to device

    # Create the model
    model = create_custom_model(len(dataset.label_encoder.classes_)).to(device)
    model.load_state_dict(torch.load(model_name))  # Load pre-trained weights

    # Register hooks to capture activations
    register_hooks(model, activations)

    # Put the model in evaluation mode
    model.eval()

    # Pass the transformed image through the model
    with torch.no_grad():
        outputs = model(transformed_image)

    # Predicted label
    _, predicted = outputs.max(1)

    # Get the class names for true and predicted labels
    true_label_text = dataset.label_encoder.inverse_transform(label.cpu())[0]
    predicted_label_text = dataset.label_encoder.inverse_transform(predicted.cpu())[0]

    # Create a new image with space below for the text
    new_height = (
        original_image.height + 50
    )  # Increase the height by 50 pixels for the text
    new_image = Image.new(
        "RGB", (original_image.width, new_height), "white"
    )  # Create a white background image
    new_image.paste(original_image, (0, 0))  # Paste the original image on top

    # Draw the true and predicted labels in the extended area (below the image)
    draw = ImageDraw.Draw(new_image)
    font = (
        ImageFont.load_default()
    )  # Load default font (you can use custom fonts if available)

    text = f"True: {true_label_text}, Predicted: {predicted_label_text}"
    text_position = (
        10,
        original_image.height + 10,
    )  # Position of the text below the image

    # Draw the text below the image
    draw.text(text_position, text, fill="black", font=font)

    # Save the new image with labels outside
    os.makedirs("activations", exist_ok=True)  # Ensure the directory exists
    new_image.save(
        f"activations/originalimage_{random_idx}.png"
    )  # Save the image with the random index

    # Save activations of the single example for visualization (optional)
    #save_activations(transformed_image, activations, epoch=0, batch_idx=random_idx,)
    save_activations(transformed_image, activations, epoch=0, batch_idx=random_idx, label_encoder=dataset.label_encoder)


    print(f"Predicted Label: {predicted_label_text}")
    print(f"True Label: {true_label_text}")
    print(
        f"\nOriginal image saved with labels outside to 'activations/originalimage_{random_idx}.png'."
    )

    # visualize kernels of selected conv layers
    # layers_to_visualize = ['conv1', 'conv2']  # Adjust layer names as needed base_model.4.0.conv1
    # layers_to_visualize = ['base_model_layer1_block1_conv1','conv1','conv2']  # Adjust layer names as needed base_model.4.0.conv1
    layers_to_visualize = [
        "base_model.4.0.conv2", # res
        "conv1",  # custom
        "conv2",
    ]
    visualize_kernels(
        model, layers_to_visualize, num_kernels=5
    )  # Pass the desired number of kernels to visualize here


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    visualize_single_example(model_name)
