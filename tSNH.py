from numpy import exp, log, sum, zeros, random

def _is_iterable(variable):
	return "__iter__" in dir(variable)

def _binary_search(f, x_min, x_max, epsilon=0, iter_max=0):
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
	epsilon: float (default=0) the error bound. If specified then the returned
	value x will only verify |f(x)| <= epsilon.
	iter_max: int (default=0) the maximum iteration. If specified then the
	returned value x will only verify |f(x)| <= c where c is unknown.
	"""
	assert type(iter_max) == int and iter_max >= 0
	
	# iteration function definition
	_is_iteration_not_finished = lambda i, n : True if n == 0 else i < n 

	iter_count = 0

	# validate x_min and x_max
	assert x_min < x_max
	
	# optimize computation
	f_min, f_max = f(x_min), f(x_max)
	
	while f_min * f_max > 0 and _is_iteration_not_finished(iter_count, iter_max):
		if f_min < f_max:
			# f is growing
			if f_min < 0:
				x_max += x_max
			else:
				x_min -= x_min
		elif f_min == f_max:
			# f is not monotonic, hope it will find the solution
			break
		else:
			# f is descending
			if f_min < 0:
				x_min -= x_min
			else:
				x_max += x_max
		f_min, f_max = f(x_min), f(x_max)
		iter_count += 1

	iter_count = 0

	# compute the binary search
	x = (x_min + x_max) / 2
	# optimize computation
	f_x, f_min, f_max = f(x), f(x_min), f(x_max)
	while abs(f_x) > epsilon and _is_iteration_not_finished(iter_count, iter_max):
		if f_x * f_min < 0:
			x_max = x
		elif f_x * f_max < 0:
			x_min = x
		else :
			return x_max if f_max == 0 else x_min
		x = (x_min + x_max) / 2
		f_x, f_min, f_max = f(x), f(x_min), f(x_max)
		iter_count += 1

	return x

def _matrix_distances_squared(X):
	"""Compute for all i the squared distances between x_i and others x_j

	Return a matrix n x n of distances squared

	Parameter:
	----------
	X : n x D array : the data 
	"""

	# euclidian distance squared
	_distance_squared = lambda x, y: sum([(x[i] - y[i])**2 for i in range(len(x))])

	n = len(X) # number of data
	d = [[0]*n for _ in range(n)] # matrix distances

	# Compute the values of matrix distances squared over/under the diagonal
	for i in range(n-1):
		for j in range(i+1, n):
			d[i][j] = _distance_squared(X[i], X[j])
			d[j][i] = d[i][j]

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
	
	# conditional probability p_j|i(beta)
	p_j_given_i = lambda beta, j, i: (exp(-beta * d_ij_2[i][j]) / sum([exp(-beta * d_ij_2[i][k]) for k in range(n) if k != i])) if j != i else 0

	# entropy of p_j|i
	H = lambda beta, i: -sum([p_j_given_i(beta, j, i) * log(p_j_given_i(beta, j, i)) if j != i and p_j_given_i(beta, j, i) > 0. else 0 for j in range(n)]) # 2nd cdt is to avoid log(0) due to exp(-746) = 0 in python :/

	p_ji_s = [[0]*n for _ in range(n)]
	
	for i in range(n):
		# H - log(Perp) = 0
		logPerp = log(Perp)
		H_logPerp_i = lambda beta: H(beta, i) - logPerp
		
		# Because argmax H(beta) = 0
		if H_logPerp_i(beta = 0) < 0:
			raise ValueError("Perplexity is too BIG!!!!")
			
		beta_i = _binary_search(H_logPerp_i, 0, n // 2 + 1)
		# sigma_2_i = 1 / (2 * beta_i)
		
		# Compute p_j|i with known sigma^2_i
		for j in range(n):
			p_ji_s[j][i] = p_j_given_i(beta_i, j, i)
	
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
	
	# joint probability q_ij
	q_ij = lambda i, j: (1 / ((1 + dy_ij_2[i][j]) * sum([sum([1 / (1 + dy_ij_2[k][l]) for k in range(n) if k != l]) for l in range(n)]))) if i != j else 0

	# Compute the value of matrix Q = q_ij for all i, j
	Q = zeros((n,n))
	for i in range(n-1):
		for j in range(i+1, n):
			Q[i][j] = q_ij(i, j)
			Q[j][i] = Q[i][j]

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
	
	# KL gradient formula
	dKL_dy_i = lambda P, Q, dy_ij_2, Y, i: 4 * sum([(P[i][j] - Q[i][j]) * (Y[i] - Y[j]) / (1 + dy_ij_2[i][j]) for j in range(n) if j != i])
	
	# Compute for all
	grad_KL = zeros((n,d))
	for i in range(n):
		grad_KL[i] = dKL_dy_i(P, Q, dy_ij_2, Y, i)
		
	return grad_KL

def tSNH(X, Perp=50, T=1000, eta=200, alpha=0., d=2):
	"""t-SNH is a method for dimension reduction of data (t-SNE done by Niaina and Hidekela).
	It is a reimplementation of t-SNE method. It aims to facilitate data visualization which
	has big dimension (greater or equal to 3).
	
	Return a matrix with d < D dimension of data from the original D dimension data.
	
	Parameters:
	-----------
	X : n x D array : the data 
	Perp: int (default=50) the perplexity
	T : int (default=1000) iteration number
	eta : int (default=200) learning rate
	alpha : float (default=0) momentum
	d : int (default=2) dimension of the reduced space
	"""
	assert _is_iterable(X) and len(X) > 0 and _is_iterable(X[0])
	assert type(Perp) == int
	assert type(T) == int and T > 0
	assert type(eta) == int
	assert type(alpha) == float
	assert type(d) == int and d > 0 and d < len(X)

	# Set variables
	n = len(X) # number of data
	d_ij_2 = _matrix_distances_squared(X) # the matrix distances squared
	
	# Compute p_{j|i} with Perp
	p_ji_s = _conditional_probabilities(X, d_ij_2, Perp)

	# Set p_ij = (p_{j|i} + p_{i|j}) / 2n
	P = zeros((n,n))
	for i in range(n-1):
		for j in range(i+1, n):
			P[i][j] = (p_ji_s[j][i] + p_ji_s[i][j]) / (2 * n)
			P[j][i] = P[i][j]
	del p_ji_s # useless from here
			
	# Sample initial solution Y = [y1, ..., yn] from N(0, 10^-4 I)
	Y = random.normal(0, 0.0001, size=(n, d))
	
	for t in range(T):
		# distances between y(s) are necessary
		dy_ij_2 = _matrix_distances_squared(Y)
		
		# Compute Q = q_ij
		Q = _reduced_dim_joint_prbabilities(Y, dy_ij_2)
		
		# Compute grad = dC/dY
		grad_KL = _grad_KL(P, Q, Y, dy_ij_2)
		
		# Set Y(t) = Y(t-1) + eta * grad + alpha * (Y(t-1) - Y(t-2))
		pass
	return Y
	
if __name__ == "__main__":
	print("Generate data...")
	X = [(0,0,1), (2,0,0), (0,3,0)]
	print("Data: ")
	print(X)
	Perp = int(input("Enter the perplexity: "))
	# ~ T = int(input("Enter the number of iterations: "))
	# ~ eta = float(input("Enter the learning rate: "))
	# ~ alpha = float(input("Enter the momentum: "))
	# ~ d = float(input("Enter the dimension d of the reduced space: "))
	tSNH(X, Perp)
	
	# Y = tSNH(X, Perp, T, eta, alpha)
	# print("The data with reduced dimension: ")
	# print(Y)

	# # BINARY SEARCH TEST
	# print("f(x) = x^2 + 2x - 1")
	# f = lambda x: x**2 + 2*x - 1
	# x = _binary_search(f, -200, -10)
	# print("x =", x, "verify f(x) = 0")
	# # x = -2.414213562373095 verify f(x) = 0 # yes it works for every interval

	# ~ # MATRIX DISTANCES SQUARED
	# ~ X = [(0,1), (1,0), (3,2), (5,9)]
	# ~ print(X)
	# ~ d_ij_2 = _matrix_distances_squared(X)
	# ~ print("The matrix distances squared is\n", d_ij_2)
	
	# ~ # CONDITIONAL PROBABILITIES TEST
	# ~ X = [(0,0,1), (2,0,0), (0,3,0)]
	# ~ print(X)
	# ~ d_ij_2 = _matrix_distances_squared(X)
	# ~ print("The matrix distances squared is\n", d_ij_2)
	# ~ Perp = int(input("Enter perplexity: "))
	# ~ p_ji_s = _conditional_probabilities(X, d_ij_2, Perp)
	# ~ print("The conditional probabilities p_ji_s are: ")
	# ~ for i in range(len(p_ji_s)):
		# ~ for j in range(len(p_ji_s[0])):
			# ~ print(f"p_{j}|{i} = {p_ji_s[j][i]}")
