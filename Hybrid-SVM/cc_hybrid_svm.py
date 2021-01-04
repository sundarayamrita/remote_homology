import csv
import argparse
from pathlib import Path
import subprocess

def pssm_extraction(filename):
    #filename=Path(filename)
    inputmat=[]
    with open(filename,'r') as f:
        lines = f.readlines()[1:]
        linesnum=len(lines)
        
        del lines[0]
        
        for line in lines:
            if 'k' in line:
                break
            elements = line.strip().split()
            temp=[]
            for i in elements:
                if(i!='' and i<'A'):
                    temp.append(i)
            if(len(temp[1:21])>1):
                inputmat.append(temp[1:21])
                        
                      
    import numpy as np  
    pssm = np.asarray(inputmat).astype(np.float32)
#    print(pssm.shape)
    pssm_col_avg = np.mean(pssm, axis=0)
    L = np.size(pssm, 0)
    N = np.size(pssm, 1)
    alpha = 3
    de2 = np.zeros((N, N - 1, alpha))

    for mu in range(alpha):
    
        for i1 in range(np.size(pssm, 1)):

            skip = False
            p_i1_avg = pssm_col_avg[i1];
            for i2 in range(np.size(pssm, 1)):
            
                p_i2_avg = pssm_col_avg[i2];
                if i1 == i2 :
                    skip = True
                    continue;
                cc = 0;
#                for j in range(L - mu):
#                    cc += (pssm[j, i1] - p_i1_avg) * (pssm[j + mu, i2][np.newaxis].T - p_i2_avg) / (L - mu)
                cc = np.matmul((pssm[ : L - mu, i1]  - p_i1_avg), (pssm[mu : L, i2][np.newaxis].T - p_i2_avg)) / (L - mu)
                if skip:
                    de2[i1, i2 - 1, mu] = cc
                else:
                    de2[i1, i2, mu] = cc
#    print(de2)

    de1 = np.zeros((N, alpha))
    for mu in range(alpha):
        for i in range(N):
            p_i_avg = pssm_col_avg[i] / (L - mu)
            de1[i, mu] = (np.matmul((pssm[ : L - mu, i]  - p_i_avg), (pssm[mu : L, i][np.newaxis].T - p_i_avg)) / (L - mu))
    print(de1)
               
#filename = Path("/home/sundarayamrita/Documents/Programming/repos/Remote-homology/PSSMs/query_8_pssm.txt")
filename = Path.cwd() / "query_1037_pssm.txt"
pssm_extraction(filename)

                





