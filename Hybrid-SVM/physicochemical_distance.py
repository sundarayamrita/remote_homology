import numpy as np
from pathlib import Path
import argparse
from normalise_aaindex import normalise  
from collections import defaultdict

def get_physico_dist(file_loc, aa_index_path):
    
    with open(file_loc, "r") as f:
        seq = f.readline()

    L = len(seq)
    alpha = 3

    lookup = {
            "A":1, "R":2, "N":3, "D":4, "C":5,
            "Q":6, "E":7, "G":8, "H":9, "I":10,
            "L":11, "K":12, "M":13, "F":14, "P":15,
            "S":16, "T":17, "W":18, "Y":19, "V":20
            }
    
    I = normalise(aa_index_path)
    num_props = I.shape[0]
    I = np.transpose(I)
    print("shape of I", I.shape)
    PDT=np.zeros((num_props,alpha))
    val = 0
    pdt = []

    for mu in range(1, alpha + 1):
        for j in range(num_props):
# AT TS WE WERE TOLD TO AVOID VARIABLES THAT ARE TO BE USED ONLY IN THE NEXT LINE SUCH AS val
# AND IF YOU CANNOT AVOID USING IT THEN GIVE IT A GOOD NAME
            val = 0
            for i in range(L - mu):
                val1 = lookup.get(seq[i], 1)
                val2 = lookup.get(seq[i + mu], 1)
                val = val + ((I[val1 - 1, j] - I[val2 - 1, j]) ** 2)
            val = val/ (L-mu)

            PDT[j, mu - 1] = val
        
        p1_idx = np.array([lookup.get(seq[i], 1) - 1 for i in range(L - mu)])
        p2_idx = np.array([lookup.get(seq[i], 1) - 1 for i in range(mu , L)])
        print("index_one",p1_idx)
        print("index_two",p2_idx)

        j = np.pad(np.arange(num_props)[:, np.newaxis], ((0,0),(0,L - mu - 1)), 'edge') #<-- 531x153
        print("j shape")
        print(j.shape)
        p1_idx = np.pad(p1_idx[np.newaxis, :], ((0, num_props - 1),(0, 0)), 'edge')
        print("p1_idx shape")
        print(p1_idx.shape)
        p2_idx = np.pad(p2_idx[np.newaxis, :], ((0, num_props - 1), (0,0)), 'edge')
        dp = np.sum(np.square(I[p1_idx, j] - I[p2_idx, j]), axis=1) / (L - mu)

        pdt.append(dp)
    
    
    print(np.asarray(pdt).T)
    print(PDT)
    
    return PDT
          
    return np.asarray(pdt).T

if __name__ == "__main__":
    file_loc = Path.cwd() / "query_1.txt"
    aa_index_path = Path.cwd() / "aaindex_format.txt"
    get_physico_dist(file_loc, aa_index_path)
