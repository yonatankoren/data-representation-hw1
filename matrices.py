import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


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
    - Recursive case: H(n) = H(1) Kronecker with H(n-1)
    
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
    
    # Recursive case: H(n) = H(1) kronecker with H(n-1)
    
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
    sign_changes = np.sum(np.diff(H, axis=1) != 0, axis=1)
    
    # Get the indices that would sort the array 'sign_changes' in ascending order.
    sign_changes_order = np.argsort(sign_changes)
    
    # Reorder the rows of H using these indices to create the Walsh-Hadamard matrix.
    H_walsh = H[sign_changes_order]
    
    return H_walsh

# Question 3 section E
def haar_matrix(n):
    """
    Generate a Haar matrix of order 2^n x 2^n using recursive construction.
    (Normalized so that rows are orthonormal).
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive natural number")
        
    # Base case: H_2
    if n == 1:
        # Normalized manually for the base case
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    # Helper to build unnormalized integer matrix
    def build_raw_haar(level):
        if level == 1:
            return np.array([[1, 1], [1, -1]])
        
        H_prev = build_raw_haar(level - 1)
        I_prev = np.eye(2**(level - 1))
        
        vec_sum = np.array([[1, 1]])
        vec_diff = np.array([[1, -1]])
        
        top_block = kronecker_product(H_prev, vec_sum)
        bottom_block = kronecker_product(I_prev, vec_diff)
        
        return np.vstack((top_block, bottom_block))

    # Build the raw matrix (integers 1, -1, 0)
    raw_haar = build_raw_haar(n)
    
    # Normalize rows
    row_norms = np.linalg.norm(raw_haar, axis=1, keepdims=True)
    
    # Divide each row by its norm
    H_normalized = raw_haar / row_norms
    # Transpose to get columns as basis functions
    H_columns_normalized = H_normalized.T
    
    return H_columns_normalized



def plot_generic_basis_functions(matrix, n, title_suffix):
    """
    Generic function to plot basis functions derived from a transformation matrix.
    
    Args:
        matrix: The transformation matrix (Hadamard, Walsh-Hadamard or Haar) size N x N.
        n: The level parameter (where N = 2^n).
        title_suffix: String to describe the specific matrix given.
    """
    N = 2**n
    
    # We transpose the matrix to access columns easily as rows for plotting
    func_values = matrix.T
    
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

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.80 + (0.01 * (6-n))) # Adjust based on density
    plt.show()




def phi_t(t):
    """ The target function phi(t) = t * exp(t) """
    return t * np.exp(t)


def get_interval_projections(t_start, t_end, N):
    """
    Computes the exact integral of phi(t) over each of the N sub-intervals using quad.
    Returns vector E where E[i] = int_{bin_i} phi(t) dt.
    """
    boundaries = np.linspace(t_start, t_end, N + 1)
    E_vec = np.zeros(N)
    
    for i in range(N):
        a, b = boundaries[i], boundaries[i+1]
        # Calculate exact integral of phi(t) on this segment
        val, _ = quad(phi_t, a, b)
        E_vec[i] = val
        
    return E_vec

def calculate_mse_integration(approx_levels, t_start, t_end):
    """
    Calculates MSE = (1/L) * int (phi(t) - approx(t))^2 dt
    using numerical integration (quad).
    """
    N = len(approx_levels)
    boundaries = np.linspace(t_start, t_end, N + 1)
    L = t_end - t_start
    total_sse = 0.0
    
    for i in range(N):
        a, b = boundaries[i], boundaries[i+1]
        level = approx_levels[i]
        
        # Define integrand for this segment: (phi(t) - constant_level)^2
        def integrand(t):
            return (phi_t(t) - level)**2
            
        # Integrate squared error
        segment_sse, _ = quad(integrand, a, b)
        total_sse += segment_sse
        
    return total_sse / L

def calculate_best_k_term_quad(basis_matrix_small, k, t_start, t_end):
    """
    Calculates best k-term approximation using integration.
    Compute Inner Products <phi, psi_j> via Integration
    The basis functions are piecewise constant.
    psi_j(t) = v_{ij} on interval i
    <phi, psi_j> = sum_i (v_{ij} * int_{interval_i} phi(t) dt)

    """
    N = basis_matrix_small.shape[0]
    
    # Get vector of integrals of phi on each bin
    E_vec_integrals = get_interval_projections(t_start, t_end, N)
    
    # Compute dot product (Matrix^T * Integrals)
    # This results in the vector of inner products <phi, psi_j>
    inner_products = basis_matrix_small.T @ E_vec_integrals
    
    # Set Coefficients directly as Inner Products
    coeffs = inner_products

    # Sort indices by the absolute value of the coefficients (descending)
    idx_sorted = np.argsort(np.abs(coeffs))[::-1]
    top_k_indices = idx_sorted[:k]
    
    # Keep only the top k coefficients, zero out the rest
    coeffs_approx = np.zeros_like(coeffs)
    coeffs_approx[top_k_indices] = coeffs[top_k_indices]
    
    # The resulting approximation is the linear combination of basis vectors
    approx_levels = basis_matrix_small @ coeffs_approx
    
    # Calculate MSE via Integration
    mse = calculate_mse_integration(approx_levels, t_start, t_end)
    
    return approx_levels, mse
    


def run_question_3g():
    print("\n--- Running Question 3g ---")
    
    n = 2
    N = 2**n
    t_start, t_end = -4, 5
    
    # Visualization Grids
    t_boundaries = np.linspace(t_start, t_end, N + 1)
    t_smooth = np.linspace(t_start, t_end, 500)
    y_smooth = phi_t(t_smooth)
    
    # Generate Bases
    H_nat = hadamard_matrix(n)
    H_walsh = walsh_hadamard_matrix(hadamard_matrix(n))
    H_haar = haar_matrix(n)
    
    bases = {
        "Hadamard": H_nat,
        "Walsh": H_walsh,
        "Haar": H_haar
    }

    # Calculate bin width dt
    dt = (t_end - t_start) / N
    
    for basis_name, matrix_raw in bases.items():

        col_norms = np.sqrt(np.sum(matrix_raw**2, axis=0) * dt)
        matrix_small = matrix_raw / col_norms

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # --- Top Plot ---
        ax1.set_title(f'Best k-term Approximations: {basis_name}')
        ax1.plot(t_smooth, y_smooth, label='$\phi(t)$', color='skyblue', linewidth=2, alpha=0.6)
        
        k_values = range(1, N + 1)
        mse_values = []
        
        print(f"\nBasis: {basis_name}")
        for k in k_values:
            # Calculate using quad integration
            approx_levels, mse = calculate_best_k_term_quad(matrix_small, k, t_start, t_end)
            mse_values.append(mse)
            print(f"  k={k}: MSE={mse:.5f}")
            
            # Simple Step Plot
            y_plot = np.append(approx_levels, approx_levels[-1])
            ax1.step(t_boundaries, y_plot, where='post', label=f'k={k}', linewidth=1.5)
            
        ax1.set_xlabel('t')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(t_start, t_end)
        
        # --- Bottom Plot ---
        ax2.set_title(f'MSE vs k ({basis_name})')
        ax2.plot(k_values, mse_values, 'o-', linewidth=2)
        ax2.set_xlabel('k')
        ax2.set_ylabel('MSE')
        ax2.set_xticks(k_values)
        ax2.grid(True)
        
        plt.tight_layout()
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

        # Generate Haar Matrix
        H_haar = haar_matrix(n)
        
        # Plot Natural Order (Question 3b)
        plot_generic_basis_functions(H_natural, n, "Hadamard (Natural Order)")
        
        # Plot Sequency Order (Question 3d)
        plot_generic_basis_functions(H_walsh, n, "Walsh-Hadamard (Sign Change Order)")

        # Plot Haar Basis Functions (Question 3f)
        plot_generic_basis_functions(H_haar, n, "Haar Basis Functions")

    # Run Question 3g
    run_question_3g()


