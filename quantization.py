import numpy as np
from PIL import Image


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

