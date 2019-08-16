import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy.spatial import distance
from random import randint


def generate_vec(n, len):
    vectors = [np.array([randint(0, 10) for j in range(len)]) for i in
               range(n)]

    data = {'num' + str(i): vectors[i] for i in range(n)}
    return DataFrame(data)


def generate_distance_matrix(n):
    distances = np.random.uniform(0, 1, int(n * (n - 1) / 2))
    data = np.zeros((n, n))
    data[np.triu_indices(n, k=1)] = distances
    data[np.tril_indices(n, k=-1)] = data.T[np.tril_indices(n, -1)]
    # np.triu_indices(n, k=1) Return the indices for the upper-triangle of an (n, m) array.
    print(data)
    return data


def normalize(df):
    result = df.copy()
    return result.apply(lambda x: x / np.sqrt(x.dot(x.T)))


def cosine_similarity(x, y):
    return x.T.dot(y)


def cosine_distance_matrix(vectors: DataFrame):
    norm_vectors = normalize(vectors)
    return 1 - cosine_similarity(norm_vectors, norm_vectors)


# example = generate_vec(10, 3)
#
# print(example)
#
# ne = normalize(example)
# print(ne)
#
# print(cosine_similarity(np.array([1 / 2, 56]), np.array([3, 0])))
# print(cosine_similarity(ne, ne))

generate_distance_matrix(5)
