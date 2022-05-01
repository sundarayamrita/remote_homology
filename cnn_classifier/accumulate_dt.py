import argparse
from pathlib import Path

from numpy.core.fromnumeric import shape
from ac_correlation import auto_correlation, cross_correlation
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

def distance_transforms(pssm_dir, seq_dir):

    alpha = 10
    filename = str(seq_dir)
    list_files = []
    aaindex_path = Path.cwd() / "aaindex_format.txt"

    acc_all = []
    pssm_dir = Path(pssm_dir)
    
    for pssm_file in pssm_dir.iterdir():
        print(pssm_file.stem)
        f = seq_dir/(pssm_file.stem[:len(pssm_file.stem)-5]+'.txt')
        with open(f, "r") as f:
            seq = f.readline()
        if len(seq)<=10:
            continue
        list_files.append(pssm_file.stem)
        de1 = auto_correlation(pssm_file)
        print("the de1 shape is:",shape(de1))
        
        print("the de1 shape is:",shape(de1))
        de2 = cross_correlation(pssm_file)
        print("the de2 shape is:",shape(de2))
        
        acc = np.hstack((de2, de1.reshape((20, 1, alpha))))
        acc_all.append(acc.reshape(-1, acc.shape[-1]))

    acc_all = np.asarray(acc_all, dtype = np.float32)

    pdt_all = []
    for seq_file in seq_dir.iterdir():
        seq = seq_file.stem + "_pssm"
        if seq in list_files:
            phsyico_dist_mat = get_physico_dist(seq_file, aaindex_path)
            if phsyico_dist_mat!= []:
                    pdt_all.append(phsyico_dist_mat)
        print("the pdt shape",shape(get_physico_dist(seq_file, aaindex_path)))
    pdt_all = np.asarray(pdt_all, dtype = np.float32)
    #print("The acc  matrix is:\n",shape(acc_all))
    #print("The  pdt matrix is: \n",shape(pdt_all))
    converted_dist = np.hstack((acc_all, pdt_all))
    #converted_dist = acc_all
    print("acc + pdt shape", converted_dist.shape)
    print("the acc pdt matrix is:",converted_dist)
    with open((filename + '_corr.npy'), 'wb') as f:
            np.save(f, converted_dist)





def standalone(seqs_dir, pssm_dir):

    list_files = []
    aaindex_path = Path.cwd() / "aaindex_format.txt"
#     filename = "TRIALS"
    for pssm_file in pssm_dir.iterdir():
        distance_transforms(pssm_dir, seqs_dir)
        list_files.append(pssm_file.stem)
    
    pdt_all = []
    for seq_file in seqs_dir.iterdir():
        seq = seq_file.stem + "_pssm"
        if seq in list_files:
            pdt_all.append(get_physico_dist(seq_file, aaindex_path))

    pdt_all = np.asarray(pdt_all, dtype = np.float32)
    print("standalone done", pdt_all.shape)
#     distance_transforms(pssm_dir, seqs_dir) 

    



if __name__ == "__main__":
   #standalone(Path(r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\Hybrid-SVM\pos-train.c.1.1"),Path(r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\Hybrid-SVM\pos-train.c.1.1PSSMs"))
   #distance_transforms(Path("/home/sundarayamrita/Documents/Programming/repos/remote_homology/acc_pdt_svm/indexed_files/a.60.1/pos-test.a.60.1PSSMs"), Path("/home/sundarayamrita/Documents/Programming/repos/remote_homology/acc_pdt_svm/indexed_files/a.60.1/pos-test.a.60.1"))
    distance_transforms(Path(r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\cnn_classifier\indexed_files\train_binary\train_binary_neg_PSSMs"),Path(r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\cnn_classifier\indexed_files\train_binary\train_binary_neg"))
