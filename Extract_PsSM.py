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
    



#pssm_extraction(r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\positive_train_pssm_new\query_1037_pssm.txt")






