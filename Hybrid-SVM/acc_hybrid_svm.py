import csv
import argparse
from pathlib import Path
import subprocess
import numpy as np
#from Remote-homology.prodec import Extract_PsSM as get_pssm

def pssm_extraction(filename):
    #filename=Path(filename)
    inputmat=[]
    with open(filename,'r') as f:
        lines = f.readlines()[1:]
        linesnum=len(lines)
        
        del lines[0]
        
        for i in range(len(lines)):
            line=lines[i]
            elements = line.strip().split()
            temp=[]
            if('Lambda' in line):
                break
            for i in elements:
                
                if(i!='' and i<'A'):
                    temp.append(i)
            if(len(temp[1:21])>1):
                inputmat.append(temp[1:21])
                        
    return(inputmat)        

def calc_de2(filename):

    pssm = np.asarray(pssm_extraction(filename)).astype(np.float32)
#    print(pssm.shape)
    pssm_row_avg = np.mean(pssm, axis=0)
    L = np.size(pssm, 0)
    N = np.size(pssm, 1)
    alpha = 3
    de2 = np.zeros((N - 1, N, alpha))

    for mu in range(alpha):

        for i1 in range(np.size(pssm, 1)):

            skip = False
            p_i1_avg = pssm_row_avg[i1];
            for i2 in range(np.size(pssm, 1)):

                p_i2_avg = pssm_row_avg[i2];
                if i1 == i2 :
                    skip = True
                    continue
                cc = 0
#                for j in range(L - mu):
#                    cc += (pssm[j, i1] - p_i1_avg) * (pssm[j + mu, i2][np.newaxis].T - p_i2_avg) / (L - mu)
                x1 = pssm[: L - mu, i1] - p_i1_avg
                x2 = pssm[mu: L, i2][np.newaxis].T - p_i2_avg
                cc = np.matmul(x1, x2) / (L - mu)
#                cc = np.matmul((pssm[ : L - mu, i1]  - p_i1_avg), (pssm[mu : L, i2][np.newaxis].T - p_i2_avg)) / (L - mu)
                if skip:
                    de2[i2 - 1, i1, mu] = cc
                else:
                    de2[i2, i1, mu] = cc
    return de2
#    print(de2)

def calc_de1(filename):

    pssm = np.asarray(pssm_extraction(filename)).astype(np.float32)
    L = np.size(pssm, 0)
    N = np.size(pssm, 1)
    alpha = 3

    pssm_row_avg = np.mean(pssm, axis=0)
    de1 = np.zeros((N, alpha))
    for mu in range(alpha):
        for i in range(N):
            p_i_avg = pssm_row_avg[i] / (L - mu)
            x1 = pssm[: L - mu, i] - p_i_avg
            x2 = pssm[mu: L, i][np.newaxis].T - p_i_avg
            de1[i, mu] = np.matmul(x1, x2) / (L - mu)
#            de1[i, mu] = (np.matmul((pssm[ : L - mu, i]  - p_i_avg), (pssm[mu : L, i][np.newaxis].T - p_i_avg)) / (L - mu))
    return de1

#filename = Path("/home/sundarayamrita/Documents/Programming/repos/Remote-homology/PSSMs/query_8_pssm.txt")
if __name__ == "__main__":

    filename = Path.cwd() / "query_213_pssm.txt"
    print(calc_de2(filename))
    print(calc_de1(filename).shape)
