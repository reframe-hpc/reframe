import numpy as np
from time import time


print('Numpy version:', np.__version__)

# Let's take the randomness out of random numbers (for reproducibility)
np.random.seed(0)

size = 4096
A, B = np.random.random((size, size)), np.random.random((size, size))
C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
E = np.random.random((int(size / 2), int(size / 4)))
F = np.random.random((int(size / 2), int(size / 2)))
F = np.dot(F, F.T)
G = np.random.random((int(size / 2), int(size / 2)))
I = (F + F.T) / 2.

# Matrix multiplication
N = 20
t = time()
for i in range(N):
    np.dot(A, B)

delta = time() - t
print('Dotted two %dx%d matrices in %0.2f s.' % (size, size, delta / N))
del A, B

# Singular Value Decomposition (SVD)
N = 3
t = time()
for i in range(N):
    np.linalg.svd(E, full_matrices=False)

delta = time() - t
print("SVD of a %dx%d matrix in %0.2f s." % (size / 2, size / 4, delta / N))
del E

# Cholesky Decomposition
N = 3
t = time()
for i in range(N):
    np.linalg.cholesky(F)

delta = time() - t
print("Cholesky decomposition of a %dx%d matrix in %0.2f s." % (size / 2,
                                                                size / 2,
                                                                delta / N))

# Eigendecomposition
t = time()
for i in range(N):
    np.linalg.eig(G)

delta = time() - t
print("Eigendecomposition of a %dx%d matrix in %0.2f s." % (size / 2,
                                                            size / 2,
                                                            delta / N))

# Matrix Inversion
t = time()
for i in range(N):
    np.linalg.inv(I)

delta = time() - t
print("Inversion of a %dx%d matrix in %0.2f s." % (size / 2,
                                                   size / 2,
                                                   delta / N))
