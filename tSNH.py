from numpy import exp, log, sum, zeros, random, array

def _is_iterable(variable):
    return "__iter__" in dir(variable)

def _binary_search(f, x_min, x_max, epsilon=1e-10, iter_max=1000):
    """
    Find the value of x such that f(x) = 0 using binary search.

    Note that f is supposed to be continuous and monotonic in [x_min, x_max]
    and verify f(x_min).f(x_max) < O.

    Return the value x such that f(x) = 0.

    Parameters:
    -----------
    f: a numerical function from real number to real number
    x_min: float
    x_max: float
    epsilon: float (default=1e-10) the error bound. If specified then the returned
    value x will only verify |f(x)| <= epsilon.
    iter_max: int (default=1000) the maximum iteration. If specified then the
    returned value x will only verify |f(x)| <= c where c is unknown.
    """
    assert type(iter_max) == int and iter_max >= 0
    
    # iteration function definition
    def _is_iteration_not_finished(i, n):
        return True if n == 0 else i < n 

    iter_count = 0

    # validate x_min and x_max
    assert x_min < x_max
    
    # optimize computation
    f_min, f_max = f(x_min), f(x_max)
    
    # Expand bounds if necessary
    while f_min * f_max > 0 and _is_iteration_not_finished(iter_count, iter_max):
        if abs(f_min) < abs(f_max):
            x_min = x_min - (x_max - x_min)
            f_min = f(x_min)
        else:
            x_max = x_max + (x_max - x_min)
            f_max = f(x_max)
        iter_count += 1

    iter_count = 0

    # compute the binary search
    x = (x_min + x_max) / 2
    f_x = f(x)
    
    while abs(f_x) > epsilon and _is_iteration_not_finished(iter_count, iter_max):
        if f_x * f_min < 0:
            x_max = x
            f_max = f_x
        else:
            x_min = x
            f_min = f_x
            
        x = (x_min + x_max) / 2
        f_x = f(x)
        iter_count += 1

    return x

def _matrix_distances_squared(X):
    """Compute for all i the squared distances between x_i and others x_j

    Return a matrix n x n of distances squared

    Parameter:
    ----------
    X : n x D array : the data 
    """
    n = len(X) # number of data
    d = zeros((n, n)) # matrix distances

    # Compute the values of matrix distances squared
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = sum((array(X[i]) - array(X[j])) ** 2)
            d[i][j] = dist_sq
            d[j][i] = dist_sq

    return d

def _conditional_probabilities(X, d_ij_2, Perp):
    """Compute for all j the conditional probability p_j|i such that 
    Perp = 2^H(p_j|i)
    
    Return a matrix containing p_j|i for all j and for all i
    
    Parameters:
    ----------
    X : n x D array : the data 
    d_ij_2 : n x n array : the matrix distances squared 
    Perp: int (default=50) the perplexity
    """
    
    assert type(Perp) == int and Perp > 0
    
    # Set variables
    n = len(X) # number of data
    
    def p_j_given_i(beta, i):
        """Compute all p_j|i for fixed i and beta"""
        # Compute numerator for all j
        numerators = exp(-beta * d_ij_2[i])
        numerators[i] = 0  # p_i|i = 0
        denominator = sum(numerators)
        
        if denominator == 0:
            return zeros(n)
        return numerators / denominator

    # entropy of p_j|i
    def H(beta, i):
        p_ji = p_j_given_i(beta, i)
        # Remove zeros to avoid log(0)
        non_zero_probs = p_ji[p_ji > 0]
        return -sum(non_zero_probs * log(non_zero_probs))

    p_ji_s = zeros((n, n))
    
    for i in range(n):
        logPerp = log(Perp)
        
        def H_logPerp_i(beta):
            return H(beta, i) - logPerp
        
        # Find beta using binary search
        try:
            beta_i = _binary_search(H_logPerp_i, 1e-8, 1e8, 1e-10, 1000)
        except:
            # Fallback if binary search fails
            beta_i = 1.0
            
        # Compute p_j|i with found beta_i
        p_ji_s[:, i] = p_j_given_i(beta_i, i)
    
    return p_ji_s
    
def _reduced_dim_joint_prbabilities(Y, dy_ij_2):
    """Compute the joint probabilities q_ij for Y 
    
    Return a matrix of q_ij for all j and for all i
    
    Parameters:
    -----------
    Y : n x d array : the random initial data in the reduced dimension
    dy_ij_2 : n x n array : the matrix distances squared (for Y)
    """
    
    # Set variables
    n = len(Y) # number of data
    
    # Compute the numerator for all pairs (Student-t distribution)
    numerators = 1 / (1 + dy_ij_2)
    # Set diagonal to 0
    for i in range(n):
        numerators[i][i] = 0
    
    # Compute normalization factor (sum over all pairs i != j)
    Z = sum(numerators)
    
    if Z == 0:
        return zeros((n, n))
    
    # Compute q_ij
    Q = numerators / Z
    
    return Q

def _grad_KL(P, Q, Y, dy_ij_2):
    """Compute the gradient of KL divergence between two distributions P and Q with respect of Y.
    
    Return a (matrix form) list of partial differentials of KL(P||Q)
    
    Parameters:
    -----------
    P : n x n array : the first distribution
    Q : n x n array : the second distribution
    Y : n x d array : the data in the reduced dimension
    dy_ij_2 : n x n array : the matrix distances squared (for Y)
    """
    
    # Set variables
    n = len(Y) # number of data
    d = len(Y[0])
    
    # Compute gradient
    grad_KL = zeros((n, d))
    for i in range(n):
        for j in range(n):
            if i != j:
                grad_KL[i] += 4 * (P[i][j] - Q[i][j]) * (Y[i] - Y[j]) / (1 + dy_ij_2[i][j])
        
    return grad_KL

def tSNH(X, Perp=50, T=1000, eta=200, alpha=0.5, d=2):
    """t-SNH is a method for dimension reduction of data (t-SNE done by Niaina and Hidekela).
    It is a reimplementation of t-SNE method. It aims to facilitate data visualization which
    has big dimension (greater or equal to 3).
    
    Return a matrix with d < D dimension of data from the original D dimension data.
    
    Parameters:
    -----------
    X : n x D array : the data 
    Perp: int (default=50) the perplexity
    T : int (default=1000) iteration number
    eta : float (default=200) learning rate
    alpha : float (default=0.5) momentum
    d : int (default=2) dimension of the reduced space
    """
    assert _is_iterable(X) and len(X) > 0 and _is_iterable(X[0])
    assert type(Perp) == int
    assert type(T) == int and T > 0
    assert type(eta) in (int, float)
    assert type(alpha) == float
    assert type(d) == int and d > 0 and d < len(X[0])

    # Set variables
    n = len(X) # number of data
    d_ij_2 = _matrix_distances_squared(X) # the matrix distances squared
    
    # Compute p_{j|i} with Perp
    p_ji_s = _conditional_probabilities(X, d_ij_2, Perp)

    # Set p_ij = (p_{j|i} + p_{i|j}) / 2n
    P = zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                P[i][j] = (p_ji_s[j][i] + p_ji_s[i][j]) / (2 * n)
            
    # Sample initial solution Y = [y1, ..., yn] from N(0, 10^-4 I)
    Y = random.normal(0, 0.0001, size=(n, d))
    
    # Initialize previous Y for momentum
    Y_prev = Y.copy()
    
    for t in range(T):
        # distances between y(s) are necessary
        dy_ij_2 = _matrix_distances_squared(Y)
        
        # Compute Q = q_ij
        Q = _reduced_dim_joint_prbabilities(Y, dy_ij_2)
        
        # Compute grad = dC/dY
        grad_KL = _grad_KL(P, Q, Y, dy_ij_2)
        
        # Update Y with momentum: Y(t) = Y(t-1) + eta * grad + alpha * (Y(t-1) - Y(t-2))
        Y_new = Y - eta * grad_KL + alpha * (Y - Y_prev)
        Y_prev = Y.copy()
        Y = Y_new
        
        # Simple learning rate decay
        eta = eta * 0.99 if t > T // 2 else eta
        
    return Y
    
if __name__ == "__main__":
    print("Generate data...")
    X = [(0,0,1), (2,0,0), (0,3,0)]
    print("Data: ")
    print(X)
    Perp = int(input("Enter the perplexity: "))
    Y = tSNH(X, Perp)
    print("Data with dimension reduced: ")
    print(Y)