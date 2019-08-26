import numpy as np
import scipy.sparse
import torch

a = np.array(
    [[1, 0, 0, 0, 0, 0, 6],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 3, 9, 0, 0, 0],
     [0, 2, 0, 0, 0, 4, 0],
     [0, 0, 0, 0, 0, 5, 0]]
)

a_max = a.max(axis=1)
print(a_max)
print(a.argmax(axis=1))

k = np.array([1, 0, 0, 0, 2, 4])
print(np.where(k == 0))  # (array([1, 2, 3]),)

a_sparse = scipy.sparse.csr_matrix(a)
print(type(a_sparse)) # <class 'scipy.sparse.csr.csr_matrix'>
print(a_sparse.toarray())  # equal to a

a1 = [1, 'as', 'borges', 7]
del(a1[0])
print(a1)

print('*******************')

a = torch.arange(0, 4).view(1, 4).long()
print(a, a.size(), a.type())
b = torch.randperm(16).view(-1, 1) * 3
print(b, b.size())
b = b.expand(16, 3)
print(b, b.size())
_range = torch.arange(0, 3).view(1, 3).long()
b = b + _range
print(b, b.size())

print(b.view(-1))

print('*******************')

c = np.array([0, 3, 9, 2, 1])
c_args = np.argsort(c)
print(c_args)
print(c[c_args])

print('*******************')

d = torch.Tensor(3).zero_()
print(d)

assert 1 == 2, '1 != 2'