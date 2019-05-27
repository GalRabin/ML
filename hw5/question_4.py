import numpy as np
from sklearn import metrics

from timeit import default_timer as timer




# Q4c - Dimension 231
def rational_variety(v):
	a = np.array([1])
	b = np.sqrt(2) * v
	c = np.square(v)
	temp = v[:, np.newaxis] * v[np.newaxis, :]
	d = np.sqrt(2) * temp[np.triu_indices(20, 1)]
	return np.concatenate((a, b, c, d))


# Q4a
data = np.random.rand(20000, 20)
data_dup = data.copy()

# Q4b

start_kernel = timer()
gram_mat_kernel = metrics.pairwise.polynomial_kernel(data, gamma=1, degree=2)
end_kernel = timer()
time_kernel = end_kernel - start_kernel

# Q4d

start_phi = timer()
data_mapped = np.vstack(rational_variety(data_dup[i, :]) for i in range(len(data_dup)))

# Q4e

gram_mat_mapped = data_mapped @ data_mapped.T
end_phi = timer()
time_phi = end_phi - start_phi

# Q4f

is_similar = np.all(np.isclose(gram_mat_kernel, gram_mat_mapped))
print('np.isclose test for the 2 matrices: Are matrices similiar? ' + str(is_similar))

# Q4g

print('Kernel time: %s' % time_kernel)
print('Phi mapping time: %s' % time_phi)

