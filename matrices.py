import numpy as np
import matplotlib.pyplot as plt


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


# Question 3 section C
def walsh_hadamard_matrix(H):
    """
    Converts a standard Hadamard matrix to a 
    Walsh-Hadamard matrix by sorting rows based on sign changes.
    
    Args:
        H: A Hadamard matrix of size 2^n x 2^n.
        
    Returns:
        numpy.ndarray: The Walsh-Hadamard matrix of the same size.
    """
    # Ensure input is a numpy array
    H = np.array(H)
    
    # Calculate the number of sign changes for each row.
    # np.diff(H, axis=1) computes the difference between adjacent elements.
    # If adjacent elements are identical (1, 1 or -1, -1), diff is 0.
    # If they are different (1, -1 or -1, 1), diff is non-zero.
    sign_changes = np.sum(np.diff(H, axis=1) != 0, axis=1)
    
    # Get the indices that would sort the array 'sign_changes' in ascending order.
    sign_changes_order = np.argsort(sign_changes)
    
    # Reorder the rows of H using these indices to create the Walsh-Hadamard matrix.
    H_walsh = H[sign_changes_order]
    
    return H_walsh

def plot_generic_basis_functions(matrix, n, title_suffix):
    """
    Generic function to plot basis functions derived from a transformation matrix.
    
    Args:
        matrix: The transformation matrix (Hadamard or Walsh-Hadamard) size N x N.
        n: The level parameter (where N = 2^n).
        title_suffix: String to describe the specific matrix given.
    """
    N = 2**n
    
    # According to Eq (4) and (5), the vector of functions is H.T * vector_v.
    # The i-th function corresponds to the i-th row of the resulting vector.
    # This is equivalent to the i-th row of H.T (which is the i-th column of H)
    # multiplied by the scaling factor sqrt(N).
    scaling_factor = np.sqrt(N)
    
    # We transpose the matrix to access columns easily as rows for plotting
    func_values = matrix.T * scaling_factor
    
    # Prepare time steps for plotting (0 to 1)
    t_steps = np.linspace(0, 1, N + 1)
    
    # Grid calculation for subplots
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.5 * rows), sharex=True, sharey=True)
    fig.suptitle(f'Basis Functions for n={n}: {title_suffix}', fontsize=16)
    
    axes_flat = axes.flatten()
    
    for i in range(N):
        ax = axes_flat[i]
        
        # Get values for the i-th function
        y = func_values[i, :]
        # Append last value for 'post' step plotting to close the line at t=1
        y_plot = np.append(y, y[-1])
        
        ax.step(t_steps, y_plot, where='post', color='blue', linewidth=1.5)
        
        # Labeling
        func_name = "hw" if "Walsh" in title_suffix else "h"
        ax.set_title(f'${func_name}_{{{i+1}}}(t)$', fontsize=10)
        
        # Styling
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        
        # Remove axis ticks for cleanliness except on edges
        if not ax.is_last_row():
            ax.set_xticklabels([])
        if not ax.is_first_col():
            ax.set_yticklabels([])

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.90 + (0.01 * (6-n))) # Adjust based on density
    plt.show()

# Main
if __name__ == "__main__":
    print("Generating plots for n = 2 to 6...")
    for n in range(2, 7):
        print(f"Processing n={n}...")
        
        # Generate Standard Hadamard
        H_natural = hadamard_matrix(n)
        
        # Generate Walsh-Hadamard
        H_walsh = walsh_hadamard_matrix(H_natural)
        
        # Plot Natural Order (Question 3b)
        plot_generic_basis_functions(H_natural, n, "Hadamard (Natural Order)")
        
        # Plot Sequency Order (Question 3d)
        plot_generic_basis_functions(H_walsh, n, "Walsh-Hadamard (Sign Change Order)")
