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
    K=I
    num_props = I.shape[0]
    I = np.transpose(I)
    print("shape of I", I.shape)
    lis_check2=[]
    lis_check22=[]
    val = 0
    mat=np.random.rand(20,531)
    PDT=[]
    pdt = np.zeros((num_props, alpha))
    sub2=[]
    for mu in range(1,alpha+1):
        p1_idx = np.array([lookup.get(seq[i], 1) -1 for i in range(L - mu)])
        p2_idx = np.array([lookup.get(seq[i], 1) - 1 for i in range(mu , L)])
        p1_values = list(map(I.__getitem__,[p1_idx]))
        p2_values = list(map(I.__getitem__,[p2_idx]))
        arr=np.sum((p1_values[0]-p2_values[0])**2,axis=0 )
        arr=arr/(L-mu)
        PDT.append(arr)
    PDT = np.asarray(PDT)
    PDT = np.transpose(PDT)


    # lis_check1=[]
    # lis_check11=[]
    # for mu in range(1, alpha + 1):
    #     for j in range(num_props):
    #         val = 0
    #         for i in range(L - mu):
    #             val1 = lookup.get(seq[i], 1)
    #             val2 = lookup.get(seq[i + mu], 1)
    #             lis_check1.append(I[val1-1,j])
    #             lis_check11.append(I[val2-1,j])
    #             val = val + (I[val1 - 1, j] - I[val2 - 1, j]) ** 2
    #         val = val / (L - mu)
    #         pdt[j, mu - 1] = val
    # print("Non-vect PDT:",pdt)
    print("Vect PDT:",PDT)
    
  


if __name__ == "__main__":
    file_loc = Path.cwd() / "query_1.txt"
    aa_index_path = Path.cwd() / "aaindex_format.txt"
    get_physico_dist(file_loc, aa_index_path)
