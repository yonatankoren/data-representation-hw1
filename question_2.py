import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def _load_and_validate_image(img_path, dtype=None):
    """Load and validate a grayscale image."""
    # Load the image
    img = Image.open(img_path)
        
    # Verify if the image is grayscale
    if img.mode != 'L':
        raise ValueError("Image must be grayscale")
        
    # Convert PIL image to numpy array
    if dtype is not None:
        img_array = np.array(img, dtype=dtype)
    else:
        img_array = np.array(img)
        
    # Verify dimensions (should be 512x512)
    if img_array.shape != (512, 512):
        raise ValueError(f"Image must be 512x512, got {img_array.shape}")
        
    return img_array

def get_sampled_image(image, D, sense='mse'):
    """
    Crops an image to 512x512 and subsamples it by factor D in the MSE sense.
    
    The 'MSE sense' implies that for every DxD block, we choose a single value
    that minimizes the Mean Squared Error for that block. Mathematically, 
    this optimal value is the arithmetic mean of the pixels in the block.

    Args:
        image (numpy.ndarray): Input grayscale image (2D array).
        D (int): Subsampling factor (e.g., 2, 4, 8...).
        sense (str): 'mse' for Mean Squared Error (uses mean), 
                'mad' for Mean Absolute Difference (uses median).

    Returns:
        numpy.ndarray: The subsampled image, after reconstruction to original size (512x512).
    
    Raises:
        ValueError: If the input image is smaller than 512x512.
    """
    
    # Ensure the image is a numpy array with Yonatan's function

    # Check if image dimensions are sufficient
    rows, cols = image.shape
    if rows < 512 or cols < 512:
        raise ValueError(f"Input image size ({rows}x{cols}) is smaller than the required 512x512.")

    # Crop the image to exactly 512x512 (top-left crop)
    img_cropped = image[:512, :512]
    
    # Calculate the new dimensions
    # We assume 512 is divisible by D (since D are powers of 2: 2, 4, 8... 256)
    new_h = 512 // D
    new_w = 512 // D
    
    # We reshape to (new_h, D, new_w, D) so we can operate on the 'D' axes
    img_reshaped = img_cropped.reshape(new_h, D, new_w, D)

    if sense.lower() == 'mse':
        # Optimal constant for MSE (L2 norm) is the Mean
        return img_reshaped.mean(axis=(1, 3))
    elif sense.lower() == 'mad':
        # Optimal constant for MAD (L1 norm) is the Median
        return np.median(img_reshaped, axis=(1, 3))
    else:
        raise ValueError(f"Unsupported sense: {sense}. Please use 'mse' or 'mad'.")
    

def reconstruct_image(sampled_image, D):
    """
    Reconstructs the full-size image from the subsampled version by repeating
    values D times in both dimensions (piecewise constant interpolation).
    """
    # Repeat elements D times along axis 0 (rows) and axis 1 (cols)
    return sampled_image.repeat(D, axis=0).repeat(D, axis=1)
    


def plot_error_vs_d(filepath, sense='mse'):
    """
    Calculates MSE for D = 2^1 to 2^8 and plots the result as a graph.
    
    Args:
        filepath (str): Path to the grayscale image.
        sense (str): 'mse' or 'mad'. Determines sampling method and error metric.
    """

    original = _load_and_validate_image(filepath)

    # Ensure strict 512x512 crop for valid comparisons
    if original.shape[0] < 512 or original.shape[1] < 512:
        raise ValueError("Image must be at least 512x512.")
    original = original[:512, :512]

    d_values = [2**i for i in range(1, 9)] # [2, 4, 8, ..., 256]
    error_values = []

    metric_name = "Mean Squared Error (MSE)" if sense == 'mse' else "Mean Absolute Difference (MAD)"

    print(f"Calculating {metric_name} for different D values...")
    for D in d_values:
        #Sample (using Mean for MSE or Median for MAD)
        sampled = get_sampled_image(original, D, sense=sense)

        reconstructed = reconstruct_image(sampled, D)
        
        # Calculate Error
        if sense == 'mse':
            error = np.mean((original - reconstructed) ** 2)
        else:
            error = np.mean(np.abs(original - reconstructed))
            
        error_values.append(error)
        print(f"D={D}: Error={error:.2f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(d_values, error_values, marker='o', linestyle='-', color='b')
    plt.title(f'{metric_name} vs. Subsampling Factor D')
    plt.xlabel('Subsampling Factor D')
    plt.ylabel(metric_name)
    plt.xscale('log', base=2) # Log scale makes powers of 2 evenly spaced
    plt.xticks(d_values, labels=d_values)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()


def display_sampled_images(filepath, sense='mse'):
    """
    Displays the raw subsampled images (smaller size) for D = 2^1 to 2^8.
    
    Args:
        filepath (str): Path to the grayscale image.
        sense (str): 'mse' or 'mad'.
    """
    original = _load_and_validate_image(filepath)
            
    if original.shape[0] < 512 or original.shape[1] < 512:
        raise ValueError("Image must be at least 512x512.")
    original = original[:512, :512]

    d_values = [2**i for i in range(1, 9)]
    
    # Create a figure for the grid
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Subsampled Images (Sense: {sense.upper()}) - Raw Sizes", fontsize=16)
    
    for idx, D in enumerate(d_values):
        sampled = get_sampled_image(original, D, sense=sense)
        
        # Create a 2x4 grid
        plt.subplot(2, 4, idx + 1)
        # Use nearest interpolation to see exact pixels clearly, especially for small images
        plt.imshow(sampled, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        plt.title(f"D={D}\nSize: {sampled.shape}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()


def display_reconstructed_images(filepath, sense='mse'):
    """
    Displays the reconstructed images (original size) for D = 2^1 to 2^8.
    
    Args:
        filepath (str): Path to the grayscale image.
        sense (str): 'mse' or 'mad'.
    """
    original = _load_and_validate_image(filepath)
            
    if original.shape[0] < 512 or original.shape[1] < 512:
        raise ValueError("Image must be at least 512x512.")
    original = original[:512, :512]

    d_values = [2**i for i in range(1, 9)]
    
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Reconstructed Images (Sense: {sense.upper()}) - 512x512", fontsize=16)
    
    for idx, D in enumerate(d_values):
        sampled = get_sampled_image(original, D, sense=sense)
        reconstructed = reconstruct_image(sampled, D)
        
        plt.subplot(2, 4, idx + 1)
        plt.imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
        plt.title(f"D={D}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

section_3_answer= """
We see that as D increases, the MSE increases as well. This is because larger D values mean more aggressive subsampling, leading to greater loss of detail in the image. We give to every DxD pixels block a single value (mean in case of MSE or median in case of MAD). As DxD get bigger - we stay with single value to describe more pixels, resulting in bigger loss. The piecewise constant interpolation used during reconstruction cannot fully recover the lost information, resulting in higher errors.

"""

if __name__ == "__main__":
    # For the assignment, make sure the image is at least 512x512
    img_path = '5.gif' 
    
    import os
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        print("Please set 'img_path' in the main block to a valid image file.")
    else:
        print("--- Running MSE Analysis ---")
        # Plot MSE graph
        plot_error_vs_d(img_path, sense='mse')
        
        # Show visual results for MSE
        display_sampled_images(img_path, sense='mse')
        display_reconstructed_images(img_path, sense='mse')

        print("\n--- Running MAD Analysis ---")
        # Plot MAD graph
        plot_error_vs_d(img_path, sense='mad')
        
        # Show visual results for MAD
        display_sampled_images(img_path, sense='mad')
        display_reconstructed_images(img_path, sense='mad')
