import csv

import argparse
from pathlib import Path
import subprocess

def pssm_extraction(filename):
    inputmat=[]
    with open(filename,'r') as f:
        lines = f.readlines()[1:]
        linesnum=len(lines)
        
        del lines[0]
        del lines[linesnum-7:]
        
       
        for line in lines:
            elements = line.strip().split()
            temp=[]
            for i in elements:
                if(i!='' and i<'A'):
                    temp.append(i)
            if(len(temp[1:21])>0):
                inputmat.append(temp[1:21])
                        
               
    return(inputmat)











