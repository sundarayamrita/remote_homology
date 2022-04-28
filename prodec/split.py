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

    for index, row in df.iterrows():
        query = "query_" + str(index)
        if row['GT'] == 0 and '_neg' in dataseqs_dir:
            query_filepath = dataseqs_dir / query
        else if row['GT'] == 1 and '_pos' in dataseqs_dir:
            query_filepath = dataseqs_dir / query
    
        with open(query_filepath.with_suffix('.txt'), 'w') as query_file:
            query_file.write(row['Sequence'])