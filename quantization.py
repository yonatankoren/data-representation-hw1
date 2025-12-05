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


# Question 1 section 1
def image_histogram(image_path):
    """
    Compute the histogram of a grayscale image.
    
    Takes a path to a 512x512 grayscale picture and returns a numpy array
    with 256 entries, where each entry contains the number of pixels with
    the corresponding gray value (0-255).
    
    Args:
        image_path: Path to the grayscale image file
    
    Returns:
        numpy.ndarray: Array of 256 integers representing the histogram,
                      where index i contains the count of pixels with gray value i
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded or is not grayscale
    """

    # Load and validate the image
    img_array = _load_and_validate_image(image_path)
    
    # Compute the histogram
    histogram = np.bincount(img_array.flatten(), minlength=256)

    histogram = histogram / (512 * 512)

    return histogram

# Question 1 section 2
def uniform_quantization(image_path, b):
    """
    Apply uniform quantization to a grayscale image using b bits.
    
    Splits the range [0, 256) into 2^b equal parts and quantizes each pixel
    to the middle value of its corresponding range.
    
    Args:
        image_path: Path to the grayscale image file
        b: Natural number representing the number of bits for quantization
    
    Returns:
        numpy.ndarray: Quantized image array with pixel values mapped to 
                      the middle values of quantization bins
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded, is not grayscale, or b is invalid
    """
    if not isinstance(b, int) or b < 1:
        raise ValueError("b must be a positive natural number")
    
    # Load and validate the image
    img_array = _load_and_validate_image(image_path, dtype=np.float64)

    if b > 8:
        return img_array
    
    # Compute number of quantization levels
    t = 2 ** b
    
    # Calculate bin width (range [0, 256) split into t equal parts)
    bin_width = 256.0 / t
    
    # Calculate the bin index for each pixel value
    # Each pixel value is divided by bin_width and floored to get its bin index
    bin_indices = np.floor(img_array / bin_width).astype(np.int32)
    
    # Clamp bin indices to valid range [0, t-1] (in case of edge case where pixel = 256)
    bin_indices = np.clip(bin_indices, 0, t - 1)
    
    # Calculate middle value for each bin
    # Middle of bin i: (i + 0.5) * bin_width
    quantized_values = (bin_indices + 0.5) * bin_width
    
    # Convert back to uint8 (natural numbers 0-255) for image representation
    quantized_img = quantized_values.astype(np.uint8)
    
    return quantized_img


def _plot_mse_vs_bits_general(image_path, quantization_func, title_suffix=""):
    """
    General function to plot MSE vs bits for any quantization algorithm.
    
    Args:
        image_path: Path to the grayscale image file
        quantization_func: Function that takes (image_path, b) and returns quantized image
        title_suffix: Optional suffix to add to the plot title
    
    Returns:
        None (displays a plot)
    """
    # Load original image
    original_img = _load_and_validate_image(image_path)
    original_img_float = original_img.astype(np.float64)
    
    # Calculate MSE for each b value
    b_values = list(range(1, 9))
    mse_values = []
    
    for b in b_values:
        # Quantize the image using the provided quantization function
        quantized_img = quantization_func(image_path, b)
        quantized_img_float = quantized_img.astype(np.float64)
        
        # Calculate MSE: mean of squared differences
        mse = np.mean((original_img_float - quantized_img_float) ** 2)
        mse_values.append(mse)
    
    # Plot MSE vs b
    plt.figure(figsize=(10, 6))
    plt.plot(b_values, mse_values, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Number of Bits (b)', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    title = 'MSE vs Number of Quantization Bits'
    if title_suffix:
        title += f' - {title_suffix}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(b_values)
    plt.tight_layout()
    plt.show()


def plot_mse_vs_bits_uniform(image_path):
    """
    Plot the Mean Squared Error (MSE) between the original image and quantized 
    image as a function of b (number of bits), for b = 1, ..., 8.
    Uses uniform quantization.
    
    Args:
        image_path: Path to the grayscale image file
    
    Returns:
        None (displays a plot)
    """
    _plot_mse_vs_bits_general(image_path, uniform_quantization, "Uniform Quantization")


def plot_mse_vs_bits_max_lloyd(image_path, epsilon=1e-6):
    """
    Plot the Mean Squared Error (MSE) between the original image and quantized 
    image as a function of b (number of bits), for b = 1, ..., 8.
    Uses Max-Lloyd quantization algorithm.
    
    Args:
        image_path: Path to the grayscale image file
        epsilon: Convergence tolerance for Max-Lloyd algorithm (default: 1e-6)
    
    Returns:
        None (displays a plot)
    """
    def max_lloyd_quantization_wrapper(img_path, b):
        """Wrapper function for Max-Lloyd quantization with fixed epsilon."""
        return max_lloyd_quantization(img_path, b, epsilon)
    
    _plot_mse_vs_bits_general(image_path, max_lloyd_quantization_wrapper, "Max-Lloyd Quantization")


def plot_quantization_levels(image_path):
    """
    Plot the decision and representation levels for uniform quantization.
    
    Shows a piecewise constant function where the x-axis represents input pixel
    values (0-255) and the y-axis represents the representation levels (output
    values after quantization).
    
    Args:
        image_path: Path to the grayscale image file (used to determine dtype)
    
    Returns:
        None (displays a plot)
    """
    # Create subplots for each b value
    b_values = [1, 2, 4, 8]
    num_plots = len(b_values)
    _, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]
    
    # Input values from 0 to 255
    input_values = np.arange(256)
    
    for idx, b in enumerate(b_values):
        # Compute number of quantization levels
        t = 2 ** b
        bin_width = 256.0 / t
        
        # Calculate representation levels (output values)
        # For each input value, find its bin and corresponding representation level
        bin_indices = np.floor(input_values / bin_width).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, t - 1)
        representation_levels = (bin_indices + 0.5) * bin_width
        
        # Plot the quantization function
        axes[idx].plot(input_values, representation_levels, linewidth=2)
        axes[idx].set_xlabel('Input Pixel Value', fontsize=11)
        axes[idx].set_ylabel('Representation Level', fontsize=11)
        axes[idx].set_title(f'Quantization Levels (b = {b}, {t} levels)', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 255)
        axes[idx].set_ylim(0, 256)
    
    plt.tight_layout()
    plt.show()


def plot_quantization_levels_max_lloyd(image_path, b_values=None, epsilon=1e-6):
    """
    Plot the decision and representation levels for Max-Lloyd quantization.
    
    Shows a piecewise constant function where the x-axis represents input pixel
    values (0-255) and the y-axis represents the representation levels (output
    values after quantization) returned from the Max-Lloyd algorithm.
    
    Args:
        image_path: Path to the grayscale image file
        b_values: List of b values to plot. If None, uses [1, 2, 4, 8]
        epsilon: Convergence tolerance for Max-Lloyd algorithm (default: 1e-6)
    
    Returns:
        None (displays a plot)
    """
    if b_values is None:
        b_values = [1, 2, 4, 8]
    
    # Get histogram PDF from the image
    histogram_pdf = image_histogram(image_path)
    
    # Create subplots for each b value
    num_plots = len(b_values)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]
    
    # Input values from 0 to 255
    input_values = np.arange(256)
    
    for idx, b in enumerate(b_values):
        # Compute number of quantization levels
        t = 2 ** b
        
        # Create uniform initial decision levels
        bin_width = 256.0 / t
        initial_decision_levels = np.array([i * bin_width for i in range(t + 1)], dtype=np.float64)
        initial_decision_levels[-1] = 256.0
        
        # Run Max-Lloyd algorithm
        decision_levels, representation_levels = max_lloyd_algorithm(
            histogram_pdf, initial_decision_levels, epsilon
        )
        
        # Calculate representation levels for each input value
        # For each input pixel value, find which bin it belongs to
        output_levels = np.zeros(256)
        
        # Vectorized approach: assign representation levels based on decision boundaries
        for i in range(len(representation_levels)):
            lower_bound = decision_levels[i]
            upper_bound = decision_levels[i + 1]
            
            # Find pixels in this bin
            if i == len(representation_levels) - 1:
                # Last bin includes upper bound
                mask = (input_values >= lower_bound) & (input_values <= upper_bound)
            else:
                mask = (input_values >= lower_bound) & (input_values < upper_bound)
            
            # Assign representation level to pixels in this bin
            output_levels[mask] = representation_levels[i]
        
        # Plot the quantization function
        axes[idx].plot(input_values, output_levels, linewidth=2)
        axes[idx].set_xlabel('Input Pixel Value', fontsize=11)
        axes[idx].set_ylabel('Representation Level', fontsize=11)
        axes[idx].set_title(f'Max-Lloyd Quantization (b = {b}, {t} levels)', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 255)
        axes[idx].set_ylim(0, 256)
    
    plt.tight_layout()
    plt.show()


def max_lloyd_algorithm(histogram_pdf, initial_decision_levels, epsilon):
    """
    Implement the Max-Lloyd algorithm for optimal quantization.
    
    The algorithm iteratively optimizes decision levels and representation levels
    to minimize Mean Squared Error (MSE). It alternates between:
    1. Computing representation levels (centroids) given decision levels
    2. Updating decision levels as midpoints between representation levels
    
    Args:
        histogram_pdf: Array of length 256 representing the probability density
                      function (PDF) of pixel values. Should be normalized (sum to 1)
                      or represent counts that will be normalized internally.
        initial_decision_levels: Array of initial decision levels (boundaries).
                                First element should be 0, last should be 256.
        epsilon: Convergence tolerance. Algorithm stops when MSE improvement < epsilon.
                Must be > 0.
    
    Returns:
        tuple: (converged_decision_levels, converged_representation_levels)
               - converged_decision_levels: Final decision levels
               - converged_representation_levels: Final representation levels
                 (centroids for each quantization bin)
    
    Raises:
        ValueError: If epsilon <= 0 or if inputs are invalid
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than 0")
    
    # Normalize histogram to PDF if needed
    pdf = np.array(histogram_pdf, dtype=np.float64)
    pdf_sum = np.sum(pdf)
    if pdf_sum > 0:
        pdf = pdf / pdf_sum
    
    # Ensure PDF has 256 elements (for pixel values 0-255)
    if len(pdf) != 256:
        raise ValueError("histogram_pdf must have length 256")
    
    # Convert decision levels to numpy array
    decision_levels = np.array(initial_decision_levels, dtype=np.float64)
    num_levels = len(decision_levels) - 1
    
    if num_levels < 1:
        raise ValueError("Must have at least one quantization level")
    
    # Pixel values (0 to 255)
    pixel_values = np.arange(256)
    
    # Initialize previous MSE to a large value
    prev_mse = float('inf')
    iteration = 0
    max_iterations = 1000  # Safety limit
    
    while iteration < max_iterations:
        # Step 1: Calculate representation levels (centroids) for each bin
        representation_levels = np.zeros(num_levels)
        
        for i in range(num_levels):
            # Find pixels in bin i: decision_levels[i] <= pixel < decision_levels[i+1]
            # For discrete values, we use: floor(decision_levels[i]) <= pixel < ceil(decision_levels[i+1])
            # Find pixels in bin i: decision_levels[i] <= pixel < decision_levels[i+1]
            # For discrete values, use floor if fractional part <= 0.5, ceil otherwise
            lower_bound = int(np.floor(decision_levels[i]) if (decision_levels[i] - np.floor(decision_levels[i])) <= 0.5 else np.ceil(decision_levels[i]))
            upper_bound = int(np.floor(decision_levels[i+1]) if (decision_levels[i+1] - np.floor(decision_levels[i+1])) <= 0.5 else np.ceil(decision_levels[i+1]))
            
            # Clamp to valid range [0, 256)
            lower_bound = max(0, min(lower_bound, 255))
            upper_bound = max(1, min(upper_bound, 256))
            
            # Get pixels in this bin
            bin_mask = (pixel_values >= lower_bound) & (pixel_values <= upper_bound)
            bin_pixels = pixel_values[bin_mask]
            bin_pdf = pdf[bin_mask]
            
            # Calculate centroid (weighted mean)
            if np.sum(bin_pdf) > 0:
                representation_levels[i] = np.sum(bin_pixels * bin_pdf) / np.sum(bin_pdf)
            else:
                # If bin is empty, use midpoint of decision levels
                representation_levels[i] = (decision_levels[i] + decision_levels[i+1]) / 2.0
        
        # Step 2: Update decision levels as midpoints between representation levels
        new_decision_levels = np.zeros(len(decision_levels))
        new_decision_levels[0] = decision_levels[0]  # Keep first boundary (usually 0)
        new_decision_levels[-1] = decision_levels[-1]  # Keep last boundary (usually 256)
        
        for i in range(1, num_levels):
            new_decision_levels[i] = (representation_levels[i-1] + representation_levels[i]) / 2.0
        
        # Step 3: Calculate MSE
        mse = 0.0
        for i in range(num_levels):
            lower_bound = int(np.floor(new_decision_levels[i]))
            upper_bound = int(np.ceil(new_decision_levels[i+1]))
            
            lower_bound = max(0, min(lower_bound, 255))
            upper_bound = max(1, min(upper_bound, 256))
            
            bin_mask = (pixel_values >= lower_bound) & (pixel_values < upper_bound)
            bin_pixels = pixel_values[bin_mask]
            bin_pdf = pdf[bin_mask]
            
            # MSE contribution from this bin
            if np.sum(bin_pdf) > 0:
                mse += np.sum(bin_pdf * (bin_pixels - representation_levels[i]) ** 2)
        
        # Check convergence: stop if MSE improvement is less than epsilon
        mse_improvement = prev_mse - mse
        if mse_improvement < epsilon:
            break
        
        # Update for next iteration
        decision_levels = new_decision_levels
        prev_mse = mse
        iteration += 1
    
    return decision_levels, representation_levels


def max_lloyd_quantization(image_path, b, epsilon=1e-6):
    """
    Apply Max-Lloyd algorithm to quantize an image.
    
    Uses the image histogram to compute optimal quantization levels via the
    Max-Lloyd algorithm, then applies the quantization to the image.
    
    Args:
        image_path: Path to the grayscale image file
        b: Natural number representing the number of bits for quantization
        epsilon: Convergence tolerance for Max-Lloyd algorithm (default: 1e-6)
    
    Returns:
        numpy.ndarray: Quantized image array with pixel values mapped to 
                      optimal representation levels
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded, is not grayscale, or b is invalid
    """
    if not isinstance(b, int) or b < 1:
        raise ValueError("b must be a positive natural number")
    
    if b > 8:
        # If b > 8, return original image (no quantization needed)
        return _load_and_validate_image(image_path)
    
    # Get histogram PDF from the image
    histogram_pdf = image_histogram(image_path)
    
    # Create uniform initial decision levels for b bits
    # Split [0, 256) into 2^b equal parts
    t = 2 ** b
    bin_width = 256.0 / t
    initial_decision_levels = np.array([i * bin_width for i in range(t + 1)], dtype=np.float64)
    initial_decision_levels[-1] = 256.0  # Ensure last boundary is exactly 256
    
    # Run Max-Lloyd algorithm
    decision_levels, representation_levels = max_lloyd_algorithm(
        histogram_pdf, initial_decision_levels, epsilon
    )
    
    # Load the image
    img_array = _load_and_validate_image(image_path)
    
    # Quantize each pixel according to the converged decision and representation levels
    quantized_img = np.zeros_like(img_array, dtype=np.float64)
    
    for i in range(len(representation_levels)):
        # Determine pixel range for this bin
        lower_bound = decision_levels[i]
        upper_bound = decision_levels[i + 1]
        
        # Find pixels in this bin
        # For the last bin, include the upper bound (pixel == 255)
        if i == len(representation_levels) - 1:
            mask = (img_array >= lower_bound) & (img_array <= upper_bound)
        else:
            mask = (img_array >= lower_bound) & (img_array < upper_bound)
        
        # Assign representation level to pixels in this bin
        quantized_img[mask] = representation_levels[i]
    
    # Convert back to uint8 (natural numbers 0-255)
    quantized_img = quantized_img.astype(np.uint8)
    
    return quantized_img


if __name__ == "__main__":
    # Path to the image file
    image_path = "5.gif"
    
    print("Generating plots for uniform quantization...")
    # Plot MSE vs bits for uniform quantization
    plot_mse_vs_bits_uniform(image_path)
    
    # Plot quantization levels for uniform quantization
    plot_quantization_levels(image_path)
    
    print("Generating plots for Max-Lloyd quantization...")
    # Plot MSE vs bits for Max-Lloyd quantization
    plot_mse_vs_bits_max_lloyd(image_path)
    
    # Plot quantization levels for Max-Lloyd quantization
    plot_quantization_levels_max_lloyd(image_path)
    
    print("All plots generated!")

