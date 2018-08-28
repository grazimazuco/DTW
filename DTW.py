import pandas as pd
import numpy as np
from math import sqrt
from timeit import default_timer as timer

def create_matrix(dataset):
    start = timer()
    
    df = pd.read_csv('C:\Data\%s_dataset.csv' % (dataset), delimiter=";")

    v_produtos = df.index.unique()
    df_dist = pd.DataFrame(columns=df.index, index=df.index)

    for prd in v_produtos:
        for prd2 in v_produtos:
            x = np.array(df.loc[prd].values).reshape(-1, 1)
            y = np.array(df.loc[prd2].values).reshape(-1, 1)

            # Python implementation based on Eamon Keogh's algorithm for DTW
            dist = DTWDistance(x, y)
            df_dist.loc[prd][prd2] = dist

	# stores the matrix with the distance points calculated
    df_dist.to_csv("C:\Data\dtw.csv"; sep=';', encoding='utf-8')
    
    end = timer()
    print('Tempo de Execucao:')
    print(end - start)


def DTWDistance(x, y):
    s1 = x[~np.isnan(x)]
    s2 = y[~np.isnan(y)]

    DTW = np.zeros(shape=(len(s1), len(s2)))

    DTW[(0,0)] = (s1[0]-s2[0]) ** 2

    for i in range(1, len(s1)):
        DTW[(i, 0)] = DTW[(i-1, 0)] + (s1[i] - s2[0]) ** 2

    for i in range(1, len(s2)):
        DTW[(0, i)] = DTW[(0, i-1)] + (s1[0] - s2[i]) ** 2

    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[i, j] = dist + min(DTW[i - 1, j - 1], DTW[i, j - 1], DTW[i - 1, j])

    distancia = sqrt(DTW[len(s1) - 1, len(s2) - 1])
    return distancia

if __name__ == '__main__':

    create_matrix('dataset_1')
    




