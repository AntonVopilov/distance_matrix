import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy.spatial import distance

a = np.array([[1, 2, 3], [3, 4, 5]])
print(distance.pdist(a, 'cosine'))

x1, x2 = np.array([1, 2, 3]), np.array([3, 4, 5]).T
print(x1)
vecs = DataFrame({'1': x1, '2': x2})

print(vecs)
print(vecs['1'])
print(vecs.shape)

vecs.to_csv('vectors.csv', sep='\t')
print()
#
new_vecs = pd.read_csv('genes_vec_8grams_1066_genes.tsv', sep='\t', header=None)

a = np.array(new_vecs.iloc[0])
print(a.dot(a.T))
s =0
for i in a:
    s+= i**2
print(s)
# new_vecs = pd.read_csv('meta_1066genes_8grams_4preds.tsv', sep='\t')
# print(new_vecs.iloc[0])