import numpy as np
import pandas as pd
from pandas import DataFrame


def generate_vec(n, dimension):
    """return n vectors as Pandas DataFrame"""
    vectors = np.random.rand(n, dimension)

    data = {'num' + str(i): vectors[i] for i in range(n)}
    return DataFrame(data)


def generate_distance_matrix(n):
    """return random distance matrix"""
    distances = np.random.uniform(0, 1, int(n * (n - 1) / 2))
    data = np.zeros((n, n))
    data[np.triu_indices(n, k=1)] = distances
    data[np.tril_indices(n, k=-1)] = data.T[np.tril_indices(n, -1)]
    return data


def normalize(df):
    result = df.copy()
    return result.apply(lambda x: x / np.sqrt(x.dot(x.T)))


def cosine_similarity(x, y):
    return x.T.dot(y)


def cosine_distance_matrix(vectors: DataFrame):
    norm_vectors = normalize(vectors)
    matrix = 1 - cosine_similarity(norm_vectors, norm_vectors)
    del norm_vectors
    return matrix


if __name__ == '__main__':


    example = generate_vec(1000, 300)

    print(example)

    print(cosine_distance_matrix(example))


