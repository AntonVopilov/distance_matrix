import pandas as pd
import cosine_distance



genes = pd.read_csv('meta_1066genes_8grams_4preds.tsv', sep='\t')

genes_names = [f'{org} {name}' for org, name in
               zip(genes['org_id_name'], genes['gene_name'])]

vectors = pd.read_csv('genes_vec_8grams_1066_genes.tsv', sep='\t', header=None)
vectors = vectors.T
vectors.columns = genes_names

matrix = cosine_distance.cosine_distance_matrix(vectors)
print(matrix)
print(matrix.iloc[-1, -1])
matrix.to_csv('distance_matrix.csv', sep='\t')