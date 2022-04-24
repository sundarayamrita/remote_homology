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

    #dataseqs_dir --> ./indexed_files/sf_index/Path(args.seq_file).stem <-- need to add neg/pos
    dataseqs_dir_pos = dataseqs_dir.parent / (dataseqs_dir.name + '_pos')
    dataseqs_dir_neg = dataseqs_dir.parent / (dataseqs_dir.name + '_neg')
    
    Path.mkdir(dataseqs_dir_pos, parents=True, exist_ok=True)
    Path.mkdir(dataseqs_dir_neg, parents=True, exist_ok=True)

    df = pd.read_csv(filepath)

    for index, row in df.iterrows():
        query = "query_" + str(index)
        if row['GT'] == 0:
            query_filepath = dataseqs_dir_neg / query
        else:
            query_filepath = dataseqs_dir_pos / query
    
        with open(query_filepath.with_suffix('.txt'), 'w') as query_file:
            query_file.write(row['Sequence'])