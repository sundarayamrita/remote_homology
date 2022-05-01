from Bio import SeqIO
from pathlib import Path
import argparse
import pandas as pd

def splitting(filepath, dataseqs):
    
    f = open(filepath,"r")
    records = list(SeqIO.parse(f, "fasta"))
    l  = len(records)
    f.close()
    
    for i in range(1, l + 1, 1):
        query = "query_" + str(i)
        file_name = dataseqs / query
    
        with open(file_name.with_suffix('.txt'), 'w') as query_file:
            query_file.write(str(records[i-1].seq))

def splitting_csv(filepath, dataseqs_dir):
    
    df = pd.read_csv(filepath)
    if "_pos" in dataseqs_dir.name:
        df = df[(df['GT']==1)]
    else:
        df = df[(df["GT"]==0)]
        
    for index, row in df.iterrows():

        query = "query_" + str(index)
        query_filepath = dataseqs_dir / query
        print("the file",query_filepath)
        with open(query_filepath.with_suffix('.txt'), 'w') as query_file:
            query_file.write(row['Sequence'])

        