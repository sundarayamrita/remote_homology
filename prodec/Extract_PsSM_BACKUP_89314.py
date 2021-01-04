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
        
<<<<<<< HEAD:prodec/Extract_PsSM.py
        for line in lines:
            if 'k' in line:
                break
=======
        for i in range(len(lines)):
            line=lines[i]
>>>>>>> 023c6c9ecae2b5178a80d2da93f1e5107c5d4122:Extract_PsSM.py
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






