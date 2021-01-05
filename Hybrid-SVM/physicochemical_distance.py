import numpy as np
import pathlib as Path
import normalise_aaindex


alpha=3

file_loc=(r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\Data_single_seq\query_1.txt")
f=open(file_loc,"r")
seq=f.readline()
print(seq)
L=len(seq)
lookup = {"A":1,"R":2,"N":3,"D":4,"C":5,
        "Q":6,"E":7,"G":8,"H":9,"I":10,
        "L":11,"K":12,"M":13,"F":14,"P":15,
        "S":16,"T":17,"W":18,"Y":19,"V":20
        }
aa_indexPath=(r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\Hybrid-SVM\aaindex_format.txt")
I=normalise_aaindex.normalise_aaindex(aa_indexPath)
I=np.transpose(I)
PDT=np.zeros((531,alpha))
val=0
for mu in range(1,alpha+1,1):
    for j in range(531):
        val=0
        for i in range(L-mu):
            val1=lookup[seq[i]]
            val2=lookup[seq[i+mu]]
            val=val+(I[val1-1,j]-I[val2-1,j])**2
            val=val/(L-mu)
        PDT[j,mu-1]=val

print(PDT.shape)

