import numpy as np


def kronecker_product(matrix_a, matrix_b):
    """
    Perform the Kronecker product between two matrices.
    
    Args:
        matrix_a: First input matrix (numpy array or list of lists)
        matrix_b: Second input matrix (numpy array or list of lists)
    
    Returns:
        numpy.ndarray: The Kronecker product of matrix_a and matrix_b
    """
    # Convert to numpy arrays if they're not already
    a = np.array(matrix_a)
    b = np.array(matrix_b)
    
    # Get dimensions
    m, n = a.shape  # matrix_a dimensions
    p, q = b.shape  # matrix_b dimensions
    
    # Initialize result matrix: (m*p) × (n*q)
    result = np.zeros((m * p, n * q), dtype=a.dtype)
    
    # Perform Kronecker product manually
    # For each element a[i][j], multiply it by the entire matrix B
    # and place it in the appropriate block position
    for i in range(m):
        for j in range(n):
            # Calculate the block position in the result matrix
            row_start = i * p
            row_end = (i + 1) * p
            col_start = j * q
            col_end = (j + 1) * q
            
            # Multiply element a[i][j] by entire matrix B
            result[row_start:row_end, col_start:col_end] = a[i, j] * b
    
    return result


def hadamard_matrix(n):
    """
    Generate a Hadamard matrix of order 2^n × 2^n using recursive construction
    with Kronecker product.
    
    A Hadamard matrix is a square matrix with entries ±1 whose rows are 
    mutually orthogonal. The recursive construction uses:
    - Base case: H(1) = [[1, 1], [1, -1]]
    - Recursive case: H(n) = H(1) ⊗ H(n-1)
    
    Args:
        n: Positive natural number representing the order (2^n × 2^n)
    
    Returns:
        numpy.ndarray: A Hadamard matrix of size (2^n) × (2^n)
    
    Raises:
        ValueError: If n is not a positive natural number
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive natural number")
    
    h1 = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    
    # Base case: H(1) is the 2×2 Hadamard matrix
    if n == 1:
        return h1
    
    # Recursive case: H(n) = H(1) ⊗ H(n-1)
    
    h_prev = hadamard_matrix(n - 1)
    
    return kronecker_product(h1, h_prev)

