import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Dictionary to store activations
# activations = {}

import os


def get_layer(model, layer_name):
    """Helper function to get layer from model, including nested layers."""
    parts = layer_name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def visualize_kernels(model, layer_names, num_kernels=5):
    ## einzelne Ornder
    
    for layer_name in layer_names:
        try:
            # Access the layer by name, including nested layers
            layer = get_layer(model, layer_name)

            if isinstance(layer, torch.nn.Conv2d):  # Ensure it's a Conv2d layer
                kernels = layer.weight.data  # Shape: (out_channels, in_channels, height, width)
                num_to_visualize = min(num_kernels, kernels.shape[0])  # Number of kernels to visualize
                
                # Create a directory for saving kernels of the current layer
                layer_dir = os.path.join("activations", f"kernels_{layer_name.replace('.', '_')}")
                os.makedirs(layer_dir, exist_ok=True)
                
                for i in range(num_to_visualize):
                    # Normalize kernel values for visualization
                    kernel = kernels[i].cpu().numpy()
                    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())  # Normalize to [0, 1]
                    
                    # Create a filename for the kernel
                    kernel_file_path = os.path.join(layer_dir, f"kernel_{i + 1}.png")
                    
                    if kernel.shape[0] == 3:  # For RGB kernels
                        # Convert kernel to HWC format and save it as an image
                        plt.imshow(np.transpose(kernel, (1, 2, 0)))
                    else:
                        # For single-channel kernels, visualize the first channel
                        plt.imshow(kernel[0], cmap="viridis")
                    
                    plt.axis("off")
                    plt.savefig(kernel_file_path, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    print(f"Saved kernel {i + 1} from layer {layer_name} to {kernel_file_path}")
            else:
                print(f"{layer_name} is not a Conv2d layer. Skipping.")
        except AttributeError:
            print(f"Could not find layer {layer_name}. Skipping.")

def save_activations(inputs, activations, epoch, batch_idx, num_samples=5, label_encoder=None):
    #save to files instead of folders, like the new one does
    # Create a directory to save activations if it doesn't exist
    if not os.path.exists("activations"):
        os.makedirs("activations")

    for layer_name, activation in activations.items():
        if layer_name == 'fc':
            # Apply softmax to the activation to get probabilities from logits
            probabilities = F.softmax(activation, dim=1)
            probabilities_np = probabilities.detach().cpu().numpy()

            # Select the first image in the batch (index 0) to visualize its probabilities
            single_image_probs = probabilities_np[0]  # Shape: (num_classes,)

            # Get class names for x-ticks
            class_names = label_encoder.inverse_transform(np.arange(len(single_image_probs)))

            # Save probabilities as a bar plot image
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(single_image_probs)), single_image_probs, color='skyblue')
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title(f'Probabilities for FC Layer, Batch_id {batch_idx}')
            plt.xticks(range(len(single_image_probs)), class_names, rotation=45)  # Use class names as x-ticks
            plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
            plt.grid(axis='y')

            # Save the probabilities plot
            prob_image_path = f"activations/probabilities_epoch{epoch}_batch{batch_idx}_fc_layer.png"
            plt.savefig(prob_image_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"Saved probabilities image at {prob_image_path}")
            
        # Create a subdirectory for the current layer if it doesn't exist
        layer_dir = os.path.join("activations", f"layer_{layer_name}")
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)

        # Move activation to CPU and convert to numpy array
        activation_np = activation.detach().cpu().numpy()

        # Select the first image in the batch (index 0) to visualize
        single_image_activations = activation_np[
            0
        ]  # Shape: (num_channels, height, width)

        # Check if the activation has spatial dimensions (height, width)
        if len(single_image_activations.shape) == 3:  # (num_channels, height, width)
            num_to_save = min(num_samples, single_image_activations.shape[0])

            valid_channels = 0  # Track how many valid activations we actually plot
            for i in range(num_to_save):
                # Ensure it's a 2D activation map
                if len(single_image_activations[i].shape) == 2:
                    # Plot and save the activation of the i-th channel
                    plt.imshow(single_image_activations[i], cmap="viridis")
                    plt.axis("off")

                    # Save the activation with a unique filename for each channel without white edges
                    # channel_file_path = f'{layer_dir}/activation_epoch{epoch}_batch{batch_idx}_layer_{layer_name}_channel_{i+1}.png'
                    channel_file_path = f"{layer_dir}/activation_b{batch_idx}_layer_{layer_name}_channel_{i + 1}.png"
                    plt.savefig(channel_file_path, bbox_inches="tight", pad_inches=0)
                    print(
                        f"Saved activation for layer {layer_name}, channel {i + 1}, at epoch {epoch}, batch {batch_idx} in {channel_file_path}"
                    )
                    valid_channels += 1
                    plt.close()
                else:
                    print(
                        f"Skipping channel {i} with shape {single_image_activations[i].shape}"
                    )

            if valid_channels == 0:
                print(f"No valid activations to plot for layer {layer_name}")
        else:
            print(
                f"Layer {layer_name} does not have spatial dimensions for visualization."
            )


def hook_fn(module, inp, out, activations, layer_name):
    activations[layer_name] = out


def register_hooks(model, activations):
    """
    Register hooks specifically on each layer of interest with unique names.
    """
    # Register hooks with unique layer names
    # model.base_model.layer1[0].conv1.register_forward_hook(lambda mod, inp, out: hook_fn(mod, inp, out, activations, "resnet_layer1_block1_conv1"))
    model.base_model[4][0].conv1.register_forward_hook(
        lambda mod, inp, out: hook_fn(
            mod, inp, out, activations, "base_model_layer1_block1_conv1"
        )
    )
    model.base_model[4][0].relu.register_forward_hook(
        lambda mod, inp, out: hook_fn(
            mod, inp, out, activations, "base_model_block1_relu1"
        )
    )
    model.base_model[4][0].conv2.register_forward_hook(
        lambda mod, inp, out: hook_fn(
            mod, inp, out, activations, "base_model_layer1_block1_conv2"
        )
    )
    model.base_model[4][0].relu.register_forward_hook(
        lambda mod, inp, out: hook_fn(
            mod, inp, out, activations, "base_model_block1_relu2"
        )
    )
    model.conv1.register_forward_hook(
        lambda mod, inp, out: hook_fn(mod, inp, out, activations, "conv1")
    )
    model.conv2.register_forward_hook(
        lambda mod, inp, out: hook_fn(mod, inp, out, activations, "conv2")
    )
    model.relu1.register_forward_hook(
        lambda mod, inp, out: hook_fn(mod, inp, out, activations, "relu1")
    )
    model.relu2.register_forward_hook(
        lambda mod, inp, out: hook_fn(mod, inp, out, activations, "relu2")
    )
    model.avgpool.register_forward_hook(
        lambda mod, inp, out: hook_fn(mod, inp, out, activations, "avgpool")
    )
    model.fc.register_forward_hook(
        lambda mod, inp, out: hook_fn(mod, inp, out, activations, "fc")
    )

"""
def save_activations_old(inputs, activations, epoch, batch_idx, num_samples=5):
    #einzelne Dateien
    
    # Create a directory to save activations if it doesn't exist
    if not os.path.exists("activations"):
        os.makedirs("activations")

    for layer_name, activation in activations.items():
        # Move activation to CPU and convert to numpy array
        activation_np = activation.detach().cpu().numpy()

        # Select the first image in the batch (index 0) to visualize
        single_image_activations = activation_np[
            0
        ]  # Shape: (num_channels, height, width)

        # Check if the activation has spatial dimensions (height, width)
        if len(single_image_activations.shape) == 3:  # (num_channels, height, width)
            num_to_save = min(num_samples, single_image_activations.shape[0])

            # Create a figure to plot the activations
            fig, axes = plt.subplots(1, num_to_save, figsize=(15, 5))

            valid_channels = 0  # Track how many valid activations we actually plot
            for i in range(num_to_save):
                # Plot the activation of the i-th channel
                ax = axes[i]
                if (
                    len(single_image_activations[i].shape) == 2
                ):  # Ensure it's a 2D activation map
                    ax.imshow(single_image_activations[i], cmap="viridis")
                    ax.axis("off")
                    ax.set_title(f"Channel {i + 1}")
                    valid_channels += 1
                else:
                    print(
                        f"Skipping channel {i} with shape {single_image_activations[i].shape}"
                    )

            # Only add colorbar if there are valid activations
            if valid_channels > 0:
                # Add a color bar to the right of the last plot for reference
                fig.colorbar(
                    axes[0].images[0],
                    ax=axes,
                    orientation="horizontal",
                    fraction=0.02,
                    pad=0.1,
                )

                # Save the figure using the unique layer name in the file name
                plt.suptitle(f"Epoch {epoch}, Batch {batch_idx}, Layer {layer_name}")
                plt.savefig(
                    f"activations/activation_epoch{epoch}_batch{batch_idx}_layer_{layer_name}.png"
                )
                print(
                    f"Saved activations for layer {layer_name} at epoch {epoch}, batch {batch_idx}"
                )
            else:
                print(f"No valid activations to plot for layer {layer_name}")

            plt.close()
        else:
            print(
                f"Layer {layer_name} does not have spatial dimensions for visualization."
            )
"""
"""
def visualize_kernels(model, layer_names, num_kernels=5):
    #einzelne dateien
    for layer_name in layer_names:
        try:
            # Access the layer by name, including nested layers
            layer = get_layer(model, layer_name)

            if isinstance(layer, torch.nn.Conv2d):  # Ensure it's a Conv2d layer
                kernels = (
                    layer.weight.data
                )  # Shape: (out_channels, in_channels, height, width)
                num_to_visualize = min(
                    num_kernels, kernels.shape[0]
                )  # Get the number of kernels to visualize

                # Create a grid to display the kernels
                fig, axes = plt.subplots(1, num_to_visualize, figsize=(15, 5))
                for i in range(num_to_visualize):
                    # Normalize kernel values for visualization
                    kernel = kernels[i].cpu().numpy()
                    kernel = (kernel - kernel.min()) / (
                        kernel.max() - kernel.min()
                    )  # Normalize to [0, 1]

                    if kernel.shape[0] == 3:  # For RGB kernels
                        axes[i].imshow(
                            np.transpose(kernel, (1, 2, 0))
                        )  # Change to HWC format
                    else:
                        # For multi-channel kernels, display the first channel
                        axes[i].imshow(kernel[0], cmap="viridis")

                    axes[i].axis("off")
                    axes[i].set_title(f"Kernel {i + 1}")

                plt.suptitle(f"First {num_kernels} Kernels from {layer_name}")

                # Ensure the activations directory exists
                os.makedirs("activations", exist_ok=True)

                plt.savefig(
                    f'activations/kernels_{layer_name.replace(".", "_")}.png'
                )  # Save the figure
                plt.close()
                print(f"Saved kernels visualization for layer {layer_name}")
            else:
                print(f"{layer_name} is not a Conv2d layer. Skipping.")
        except AttributeError:
            print(f"Could not find layer {layer_name}. Skipping.")

"""
