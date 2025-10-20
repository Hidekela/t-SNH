import numpy as np

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

	Return a matrix n x n of distances squared (NumPy array)

	Parameter:
	----------
	X : n x D np.array : the data 
	"""
	# Vectorized computation based on (x-y)^2 = x^2 - 2xy + y^2
	X = np.array(X)
	sum_X = np.sum(np.square(X), 1) # (n,)
	# -2xy term: np.dot(X, X.T) is the matrix product (n,D) x (D,n) -> (n,n)
	# sum_X[:, np.newaxis] broadcasts sum_X (n,) to (n,n) for the addition
	# sum_X[np.newaxis, :] broadcasts sum_X (n,) to (n,n) for the addition
	D = np.add(np.add(-2 * np.dot(X, X.T), sum_X[:, np.newaxis]), sum_X[np.newaxis, :])
	return D

def _H_i(beta, i, dist_row, n):
	"""Compute the entropy H for a given line i and beta (vectorized)
	
	Return the entropy H
	
	Parameters:
	----------
	beta : float (= 1/2sigma^2)
	i : int the line index
	dist_row : array of floats the i-th line of distances
	n : int the length of the data 
	"""
	# Using max substraction for numerical stability
	
	# Create a mask to exclude the element i (d_ii)
	mask = np.arange(n) != i
	
	betaminusD = -beta * dist_row
	
	# Max unless for i==j
	if np.any(mask):
		betaminusD_max = np.max(betaminusD[mask])
	else: # Case of n=1 (low probability but tired  getting errors :/)
		return 0.0

	P_num = np.exp(betaminusD - betaminusD_max)
	P_num[i] = 0.0 # p_i|i = 0
	sumP = np.sum(P_num)
	
	if sumP == 0.0: return 0.0
	
	P = P_num / sumP
	# Compute the entropy
	# Select only P > 0 to avoid log(0)
	P_pos = P[P > 1e-12] # for robustness 
	H = -np.sum(P_pos * np.log(P_pos))
	return H

def _get_P_col(beta, i, dist_row, n):
	"""Compute the colonne of p_j|i for a final value of beta (vectorized)
	
	Return the column of probabilities p_j|i
	
	Parameters:
	----------
	beta : float (= 1/2sigma^2)
	i : int the column index
	dist_row : array of floats the i-th line of distances
	n : int the length of the data 
	"""
	# Create a mask to exclude the element i
	mask = np.arange(n) != i
	
	betaminusD = -beta * dist_row
	
	if np.any(mask):
		betaminusD_max = np.max(betaminusD[mask])
	else:
		return np.zeros(n)

	P_num = np.exp(betaminusD - betaminusD_max)
	P_num[i] = 0.0
	sumP = np.sum(P_num)
	
	if sumP == 0.0: return np.zeros(n)
	
	return P_num / sumP

def _conditional_probabilities(X, d_ij_2, Perp):
	"""Compute for all j the conditional probability p_j|i such that 
	Perp = 2^H(p_j|i)
	
	Return a matrix containing p_j|i for all j and for all i (NumPy array)
	
	Parameters:
	----------
	X : n x D np.array : the data 
	d_ij_2 : n x n np.array : the matrix distances squared 
	Perp: int or float (default=50) the perplexity
	"""
	
	assert isinstance(Perp, (int, float)) and Perp > 0
	
	# Set variables
	n = len(X) # number of data
	
	p_ji_s = np.zeros((n, n))
	logPerp = np.log(Perp)
	
	# Compute for all i (vectorized with _H_i and get_P_col)
	for i in range(n):
		d_i = d_ij_2[i, :] # the i-th line of distances
		
		# H - log(Perp) = 0
		H_logPerp_i = lambda beta: _H_i(beta, i, d_i, n) - logPerp
		
		# Because argmax H(beta) = 0
		if H_logPerp_i(beta = 0) < 0:
			print(f"Warning: Perplexity {Perp} might be too high for n={n} points.")

		# Do not run it so much, approximation value will be enough (limit iter_max=50)
		beta_i = _binary_search(H_logPerp_i, 0, n * 2, epsilon=1e-5, iter_max=50)
		
		# Compute p_j|i with the beta_i found
		p_ji_s[:, i] = _get_P_col(beta_i, i, d_i, n)
	
	return p_ji_s

def _reduced_dim_joint_prbabilities_num(Y, dy_ij_2):
	"""Compute the unnormalized numerators (1 / (1 + dy_ij_2)) for q_ij 
	
	Return a matrix of unnormalized numerators num_q = (1 / (1 + dy_ij_2))
	
	Parameters:
	-----------
	Y : n x d np.array : the data in the reduced dimension
	dy_ij_2 : n x n np.array : the matrix distances squared (for Y)
	"""
	# Compute num_q = 1 / (1 + D_ij_2) (vectorized)
	num_q = 1 / (1 + dy_ij_2)
	
	# q_ii = 0
	np.fill_diagonal(num_q, 0.0)
	
	return num_q


def _grad_KL(P, Q, Y, num_q):
	"""Compute the gradient of KL divergence between two distributions P and Q with respect of Y (vectorisé).
	
	Return a (matrix form) np.array of partial differentials of KL(P||Q)
	
	Parameters:
	-----------
	P : n x n np.array : the first distribution
	Q : n x n np.array : the second distribution
	Y : n x d np.array : the data in the reduced dimension
	num_q : n x n np.array : the unnormalized numerator 1 / (1 + dy_ij_2)
	"""
	
	# (p_ij - q_ij) * (1 + ...)^-1 -> (n, n)
	# Extend it to (n, n, 1) for multiplying it by Y_diff (n, n, d)
	gradient_term = (P - Q) * num_q
	
	# (y_i - y_j)
	# (n, 1, d) - (1, n, d) -> (n, n, d)
	Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :] 
	
	# Complet term: (n, n, d)
	full_term = gradient_term[:, :, np.newaxis] * Y_diff
	
	# Sommation over j
	grad_KL = 4 * np.sum(full_term, axis=1) # (n, d)
		
	return grad_KL

def tSNH(X, Perp=50, T=1000, eta=200.0, alpha=0.5, d=2, random_state=None):
	"""t-SNH is a method for dimension reduction of data (t-SNE done by Niaina and Hidekela).
	It is a reimplementation of t-SNE method. It aims to facilitate data visualization which
	has big dimension (greater or equal to 3).
	
	Return a matrix with d < D dimension of data from the original D dimension data.
	
	Parameters:
	-----------
	X : n x D np.array : the data 
	Perp: int or float (default=50) the perplexity
	T : int (default=1000) iteration number
	eta : float (default=200.0) learning rate (changé en float)
	alpha : float (default=0.5) momentum (changé en 0.5)
	d : int (default=2) dimension of the reduced space
	random_state: int or None (default=None) seed for reproducibility
	"""
	assert _is_iterable(X) and len(X) > 0 and _is_iterable(X[0])
	assert isinstance(Perp, (int, float))
	assert type(T) == int and T > 0
	assert type(eta) == float
	assert type(alpha) == float
	assert type(d) == int and d > 0 

	# Set variables
	X = np.array(X, dtype=float)
	n = len(X) # number of data
	
	# Compute distances
	d_ij_2 = _matrix_distances_squared(X) # the matrix distances squared 
	
	# Compute p_{j|i} with Perp
	p_ji_s = _conditional_probabilities(X, d_ij_2, Perp) # NumPy array

	# Set p_ij = (p_{j|i} + p_{i|j}) / 2n
	P = (p_ji_s + p_ji_s.T) / (2 * n)
	
	# Optimization : "Early Exaggeration" 
	P_exaggerated = P * 12 
	
	# Ensure that P is at least > 0 to avoid log(0)
	P = np.maximum(P, 1e-12)
	P_exaggerated = np.maximum(P_exaggerated, 1e-12)

	print("Optimizing low-dimensional embedding Y...")
	
	# Init Y
	rng = np.random.RandomState(random_state)
	Y = rng.normal(0, 1e-4, (n, d)) # NumPy array
	
	# Variables pour le momentum (Étape 2.2c)
	# Y_prev est Y(t-1) et Y_prev_prev est Y(t-2)
	Y_prev = Y.copy()
	Y_prev_prev = Y.copy()
	
	for t in range(T):
		# distances between y(s) are necessary
		dy_ij_2 = _matrix_distances_squared(Y) # NumPy array
		
		# Compute Q = q_ij (2 steps)
		# 1. Compute the numerator unnormalized
		num_q = _reduced_dim_joint_prbabilities_num(Y, dy_ij_2) # 1 / (1 + dy_ij_2)
		
		# 2. Normalization
		sum_q = np.sum(num_q) # Denominator of Q
		Q = num_q / sum_q
		Q = np.maximum(Q, 1e-12) # Avoid null values
		
		# Use P_exaggerated for the 100 first iterations.
		if t < 100:
			current_P = P_exaggerated
		else:
			current_P = P
		
		# Compute grad = dC/dY (vectorized)
		grad = _grad_KL(current_P, Q, Y, num_q)
		
		# Set Y(t) = Y(t-1) - eta * grad + alpha * (Y(t-1) - Y(t-2))
		
		# (Y(t-1) - Y(t-2))
		momentum_term = alpha * (Y_prev - Y_prev_prev)
		
		# Update positions
		# Y_new = Y(t)
		Y_new = Y_prev - eta * grad + momentum_term
		
		# Update the past positions for the next itération
		Y_prev_prev = Y_prev.copy()
		Y_prev = Y_new.copy()
		Y = Y_new 
		
		 # Show progression (KL divergence)
		if t % 100 == 0 or t == T - 1:
			if t < 100: # During exagération
				cost = np.sum(P_exaggerated * np.log(P_exaggerated / Q))
			else:
				cost = np.sum(P * np.log(P / Q))
			print(f"Iteration {t}: KL Divergence = {cost:.4f}")
		
	# Finished
	print("Optimization finished.")
	return Y
	
if __name__ == "__main__":
	
	X = [(0,0,1), (2,0,0), (0,3,0), (1, 1, 1), (5, 0, 0)]
	print("X data:")
	print(np.array(X))
	
	Perp = 2.0
	T = 1000 # iterations number
	eta = 100.0 # Learning rate
	alpha = 0.5 # Momentum
	
	Y = tSNH(X, Perp, T, eta, alpha, d, random_state=42)
	
	print("\nThe matrix Y (result) is:\n", Y)
