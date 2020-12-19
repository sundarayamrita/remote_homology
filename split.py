from Bio import SeqIO
import os
import argparse
from pathlib import Path

filepath=Path.cwd()/"neg-test.c.1.1.fasta"
dataseqs=Path.cwd()/"Data_single_seq"
if not os.path.exists(dataseqs):
	os.mkdir(dataseqs)

f=open(filepath,"r")
records = list(SeqIO.parse(f, "fasta"))
l=len(records)
for i in range(1,l+1,1):
    query="query_"+str(i)
    file_name=os.path.join(dataseqs,query)
    f = open(file_name+".txt", "w")
    f.write(str(records[i-1].seq))
    f.close()

 