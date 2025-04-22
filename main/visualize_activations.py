from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import cm
import os

# Step 1: Load Activation Images
def load_activation_images(layer_dirs):
    layer_images = []
    for layer_dir in layer_dirs:
        # Load all images in the layer's directory
        images = [Image.open(os.path.join(layer_dir, file)) for file in sorted(os.listdir(layer_dir)) if file.endswith('.png')]
        layer_images.append(images)
    return layer_images

# Step 2: Create a Grid of Images (stacking vertically)
def plot_activation_grid(layer_images, layer_names, colormap='coolwarm'):
    num_layers = len(layer_images)
    max_channels = max([len(images) for images in layer_images])  # Find the max number of channels across layers

    fig, axes = plt.subplots(nrows=max_channels, ncols=num_layers, figsize=(20, 20), constrained_layout=True)

    # Plot each layer's activation maps (stacked vertically)
    for i, images in enumerate(layer_images):
        num_channels = len(images)

        for j in range(num_channels):
            ax = axes[j, i] if max_channels > 1 else axes[i]
            ax.imshow(images[j], cmap=colormap)  # Use the colormap here
            ax.axis('off')  # Hide axis for clarity

        # Add layer name at the top of the column
        axes[0, i].set_title(layer_names[i], fontsize=12, pad=20)  # Adjust padding for spacing

        # Add color bar at the bottom of the column
        fig.colorbar(cm.ScalarMappable(cmap=colormap), ax=axes[:, i], orientation='horizontal', fraction=0.02, pad=0.04)

    return fig, axes

# Step 3: Add Modified Connections Between Layers
def add_connections_between_layers(fig, axes, layer_images):
    # kernels_base_model_4_0_conv2 -> layer_base_model_layer1_block1_conv2 (1-to-1)
    for j in range(len(layer_images[0])):  # kernels_base_model_4_0_conv2 channels
        xy_curr = (1.0, 0.5)  # Right edge, middle of kernels_base_model_4_0_conv2
        xy_next = (0.0, 0.5)  # Left edge, middle of layer_base_model_layer1_block1_conv2
        con = ConnectionPatch(xyA=xy_curr, xyB=xy_next,
                              coordsA="axes fraction", coordsB="axes fraction",
                              axesA=axes[j, 0], axesB=axes[j, 1],
                              color="gray", lw=0.5, alpha=0.7)
        fig.add_artist(con)

    # layer_base_model_layer1_block1_conv2 -> layer_base_model_block1_relu2 (1-to-1)
    for j in range(len(layer_images[1])):  # layer_base_model_layer1_block1_conv2 channels
        xy_curr = (1.0, 0.5)  # Right edge, middle of layer_base_model_layer1_block1_conv2
        xy_next = (0.0, 0.5)  # Left edge, middle of layer_base_model_block1_relu2
        con = ConnectionPatch(xyA=xy_curr, xyB=xy_next,
                              coordsA="axes fraction", coordsB="axes fraction",
                              axesA=axes[j, 1], axesB=axes[j, 2],
                              color="gray", lw=0.5, alpha=0.7)
        fig.add_artist(con)

    # layer_base_model_block1_relu2 -> kernels_conv1 (all-to-all)
    for j in range(len(layer_images[2])):  # layer_base_model_block1_relu2 channels
        for k in range(len(layer_images[3])):  # kernels_conv1 channels
            xy_curr = (1.0, 0.5)  # Right edge, middle of layer_base_model_block1_relu2
            xy_next = (0.0, 0.5)  # Left edge, middle of kernels_conv1
            con = ConnectionPatch(xyA=xy_curr, xyB=xy_next,
                                  coordsA="axes fraction", coordsB="axes fraction",
                                  axesA=axes[j, 2], axesB=axes[k, 3],
                                  color="gray", lw=0.5, alpha=0.7)
            fig.add_artist(con)

    # kernels_conv1 -> layer_conv1 (1-to-1)
    for j in range(len(layer_images[3])):  # kernels_conv1 channels
        xy_curr = (1.0, 0.5)  # Right edge, middle of kernels_conv1
        xy_next = (0.0, 0.5)  # Left edge, middle of layer_conv1
        con = ConnectionPatch(xyA=xy_curr, xyB=xy_next,
                              coordsA="axes fraction", coordsB="axes fraction",
                              axesA=axes[j, 3], axesB=axes[j, 4],
                              color="gray", lw=0.5, alpha=0.7)
        fig.add_artist(con)

    # layer_conv1 -> layer_relu1 (1-to-1)
    for j in range(len(layer_images[4])):  # layer_conv1 channels
        xy_curr = (1.0, 0.5)  # Right edge, middle of layer_conv1
        xy_next = (0.0, 0.5)  # Left edge, middle of layer_relu1
        con = ConnectionPatch(xyA=xy_curr, xyB=xy_next,
                              coordsA="axes fraction", coordsB="axes fraction",
                              axesA=axes[j, 4], axesB=axes[j, 5],
                              color="gray", lw=0.5, alpha=0.7)
        fig.add_artist(con)

    # layer_relu1 -> kernels_conv2 (all-to-all)
    for j in range(len(layer_images[5])):  # layer_relu1 channels
        for k in range(len(layer_images[6])):  # kernels_conv2 channels
            xy_curr = (1.0, 0.5)  # Right edge, middle of layer_relu1
            xy_next = (0.0, 0.5)  # Left edge, middle of kernels_conv2
            con = ConnectionPatch(xyA=xy_curr, xyB=xy_next,
                                  coordsA="axes fraction", coordsB="axes fraction",
                                  axesA=axes[j, 5], axesB=axes[k, 6],
                                  color="gray", lw=0.5, alpha=0.7)
            fig.add_artist(con)

    # kernels_conv2 -> layer_conv2 (1-to-1)
    for j in range(len(layer_images[6])):  # kernels_conv2 channels
        xy_curr = (1.0, 0.5)  # Right edge, middle of kernels_conv2
        xy_next = (0.0, 0.5)  # Left edge, middle of layer_conv2
        con = ConnectionPatch(xyA=xy_curr, xyB=xy_next,
                              coordsA="axes fraction", coordsB="axes fraction",
                              axesA=axes[j, 6], axesB=axes[j, 7],
                              color="gray", lw=0.5, alpha=0.7)
        fig.add_artist(con)

    # layer_conv2 -> layer_relu2 (1-to-1)
    for j in range(len(layer_images[7])):  # layer_conv2 channels
        xy_curr = (1.0, 0.5)  # Right edge, middle of layer_conv2
        xy_next = (0.0, 0.5)  # Left edge, middle of layer_relu2
        con = ConnectionPatch(xyA=xy_curr, xyB=xy_next,
                              coordsA="axes fraction", coordsB="axes fraction",
                              axesA=axes[j, 7], axesB=axes[j, 8],
                              color="gray", lw=0.5, alpha=0.7)
        fig.add_artist(con)

# Step 5: Combine Everything to Visualize Activations
def visualize_precomputed_activations(layer_dirs, layer_names, colormap='coolwarm'):
    # Load images from directories
    layer_images = load_activation_images(layer_dirs)

    # Create the grid of activations (stacking vertically)
    fig, axes = plot_activation_grid(layer_images, layer_names, colormap=colormap)

    # Add straight connections between layers
    add_connections_between_layers(fig, axes, layer_images)

    # Show the final visualization
    plt.show()

# Example usage (assuming activations are stored in directories named by layer)
layer_dirs = [
    'activations/kernels_base_model_4_0_conv2', 'activations/layer_base_model_layer1_block1_conv2', 
    'activations/layer_base_model_block1_relu2', 'activations/kernels_conv1', 
    'activations/layer_conv1', 'activations/layer_relu1', 
    'activations/kernels_conv2', 'activations/layer_conv2', 
    'activations/layer_relu2'
]

layer_names = [
    "kernels_conv2", "resnet_conv2", 
    "resnet_relu2", "kernels_conv1", 
    "layer_conv1", "layer_relu1", 
    "kernels_conv2", "layer_conv2", 
    "layer_relu2"
]

# Call visualization with color map choice
visualize_precomputed_activations(layer_dirs, layer_names, colormap='coolwarm')