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




def auto_corr(filename):

    pssm = np.asarray(pssm_extraction(filename)).astype(np.float32)
    print("pssm shape", pssm.shape)
    L = np.size(pssm, 0)
    N = np.size(pssm, 1)
    alpha = 3

    pssm_row_avg = np.mean(pssm, axis=0)
    de1 = np.zeros((N, alpha))

    for mu in range(1,alpha+1,1):
 
        for i in range(N):
            p_i_avg = pssm_row_avg[i]
            x1 = pssm[: L - mu, i] - p_i_avg
            x2 = pssm[mu: L, i] - p_i_avg
            d = (np.matmul(x1,x1))/L
            print("the d is:",d)  
            de1[i, mu-1] = (np.matmul(x1, x2) / (L - mu))/d

    return (de1)

#filename = Path("/home/sundarayamrita/Documents/Programming/repos/Remote-homology/PSSMs/query_8_pssm.txt")
if __name__ == "__main__":

    filename = Path.cwd() / "query_213_pssm.txt"
#    print(calc_de2(filename).shape)
#    print(calc_de1(filename).shape)
    de1 = auto_corr(filename)
    print("de1", de1.shape)
    print("The de1", de1)
