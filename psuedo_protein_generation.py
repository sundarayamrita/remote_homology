import csv
import numpy as np
with open('query_12_pssm.txt','r') as f:
    inputmat=[]    
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
    for row in inputmat:
        print(row)
       
    
    