from Bio import SeqIO
import os
import argparse
from pathlib import Path

def spliting(filepath,dataseqs) :

    f = open(filepath,"r")
    records = list(SeqIO.parse(f, "fasta"))
    l  = len(records)
    for i in range(1,l+1,1):
        query = "query_"+str(i)
        file_name = os.path.join(dataseqs,query)
        f = open(file_name+".txt", "w")
        f.write(str(records[i-1].seq))
        f.close()
