import argparse
from pathlib import Path
from acc_hybrid_svm import calc_de1, calc_de2
import numpy as np
from physicochemical_distance import get_physico_dist

# aaindex_path = Path.cwd() / "aaindex_format.txt"

# parser = argparse.ArgumentParser()
# parser.add_argument("pssm_dir", help = "pssm directory path", type = str)
# parser.add_argument("seq_dir", help = "raw seq directory path", type = str)
# parser.add_argument("--aaindex_dir", nargs = '?', default = aaindex_path, help = "aaindex1 directory path", type = str)
# args = parser.parse_args()

# if args.pssm_dir:
#     pssm_dir = Path(args.pssm_dir)
# if args.seq_dir:
#     seq_dir = Path(args.seq_dir)
# if args.aaindex_dir:
#     aaindex_path = args.aaindex_dir

# acc_all = []
# for pssm_file in pssm_dir.iterdir():

#     de1 = calc_de1(pssm_file)
#     de2 = calc_de2(pssm_file)
#     #acc = np.append(de2, de1, axis=1)
#     acc = np.vstack((de2, de1.reshape((1, 20, 3))))
#     acc_all.append(acc.reshape(-1, acc.shape[-1]))

# acc_all = np.asarray(acc_all, dtype = np.float32)
# print(acc_all.shape)


# pdt_all = []
# for seq_file in seq_dir.iterdir():
#     pdt_all.append(get_physico_dist(seq_file, aaindex_path))

# pdt_all = np.asarray(pdt_all, dtype = np.float32)
# print(pdt_all.shape)

# converted_dist = np.hstack((acc_all, pdt_all))
# print(converted_dist.shape)
# with open('neg.npy', 'wb') as f:
#     np.save(f, converted_dist)

def Distance_Transforms(pssm_dir,seq_dir,filename) :

    list_files=[]
    aaindex_path = Path.cwd() / "aaindex_format.txt"

    # parser = argparse.ArgumentParser()
    # parser.add_argument("pssm_dir", help = "pssm directory path", type = str)
    # parser.add_argument("seq_dir", help = "raw seq directory path", type = str)
    # parser.add_argument("--aaindex_dir", nargs = '?', default = aaindex_path, help = "aaindex1 directory path", type = str)
    # args = parser.parse_args()

    # if args.aaindex_dir:
    #     aaindex_path = args.aaindex_dir

    acc_all = []
   
    pssm_dir = Path(pssm_dir)
   
    
    for pssm_file in pssm_dir.iterdir():
        list_files.append(pssm_file.stem)
        de1 = calc_de1(pssm_file)
        de2 = calc_de2(pssm_file)
    #acc = np.append(de2, de1, axis=1)
        acc = np.vstack((de2, de1.reshape((1, 20, 3))))
        acc_all.append(acc.reshape(-1, acc.shape[-1]))

    acc_all = np.asarray(acc_all, dtype = np.float32)
    print(acc_all.shape)


    pdt_all = []
    for seq_file in seq_dir.iterdir():
        seq=seq_file.stem + "_pssm"
        if seq in list_files:
        
            pdt_all.append(get_physico_dist(seq_file, aaindex_path))

    pdt_all = np.asarray(pdt_all, dtype = np.float32)
    print(pdt_all.shape)

    converted_dist = np.hstack((acc_all, pdt_all))
    print(converted_dist.shape)
    with open((filename+'.npy'), 'wb') as f:
            np.save(f, converted_dist)





def checking_trial(seqs_dir,pssm_dir):
    list_files=[]
    aaindex_path = Path.cwd() / "aaindex_format.txt"
    filename="tRIALS"
    for pssm_file in pssm_dir.iterdir():
        list_files.append(pssm_file.stem)
    
    pdt_all = []
    for seq_file in seqs_dir.iterdir():
        seq=seq_file.stem + "_pssm"
        if seq in list_files:
        
            pdt_all.append(get_physico_dist(seq_file, aaindex_path))

    pdt_all = np.asarray(pdt_all, dtype = np.float32)
    print(pdt_all.shape)

    



if __name__=="__main__":
   checking_trial(Path(r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\Hybrid-SVM\pos-train.c.1.1"),Path(r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\Hybrid-SVM\pos-train.c.1.1PSSMs"))