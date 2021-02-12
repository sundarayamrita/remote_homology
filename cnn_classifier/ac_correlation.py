import csv
import argparse
from pathlib import Path
import subprocess
import numpy as np

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


def auto_correlation(filename):

    pssm = np.asarray(pssm_extraction(filename)).astype(np.float32)
    L = np.size(pssm, 0)
    N = np.size(pssm, 1)
    alpha = 10

    pssm_row_avg = np.mean(pssm, axis=0)
    de1 = np.zeros((N, alpha))

    for mu in range(1,alpha+1,1):
 
        for i in range(N):

            p_i_avg = pssm_row_avg[i]
            x1 = pssm[: L - mu, i] - p_i_avg
            x2 = pssm[mu: L, i] - p_i_avg
            d = (np.matmul(x1,x1))/L
            de1[i, mu-1] = (np.matmul(x1, x2) / (L - mu))/d

    return (de1)

def cross_correlation(filename):

    pssm = np.asarray(pssm_extraction(filename)).astype(np.float32)
    pssm_aa_avg = np.mean(pssm, axis = 0)
    L = np.size(pssm, 0)
    N = np.size(pssm, 1)
    alpha = 10
    de2 = np.zeros((N, N - 1, alpha))

    for mu in range(alpha):

        for i in range(N):

            x1 = pssm[: L - mu, i] - pssm_aa_avg[i]
            x2 = np.delete(pssm[mu: L], i, 1) - pssm_aa_avg[i]
            numerator = np.matmul(x1, x2) / L
            denominator = np.mean(np.square(np.delete(pssm[: L - mu], i, 1) - pssm_aa_avg[i]), axis = 0) / L
            de2[i, :, mu] = numerator / denominator

    return de2

if __name__ == "__main__":

    filename = Path.cwd() / "query_213_pssm.txt"
    print("auto correlation", auto_correlation(filename).shape)
    print("cross correlation", cross_correlation(filename).shape)
