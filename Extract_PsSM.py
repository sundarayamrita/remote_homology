import csv
import numpy as np
import argparse
from pathlib import Path
import subprocess

def pssm_extraction(file_name):
    PSSM_dir=Path(Path.joinpath(Path.cwd(),"PSSMs"))
    print(PSSM_dir)
    inputmat=[]
    for files in PSSM_dir.iterdir():
        files=str(files)
        with open(files,'r') as f:   
            lines = f.readlines()[1:]
            linesnum=len(lines)
            del lines[0]
            del lines[linesnum-9:]
            for line in lines:
                elements = line.strip().split()
                temp=[]
                for i in elements:
                    if(i!='' and i<'A'): 
                        temp.append(i)
                if(len(temp)>0):
                    inputmat.append(temp[21:41])

        return(inputmat)


pssm_extraction()
    
