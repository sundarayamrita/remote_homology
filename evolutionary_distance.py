import Extract_Pssm_hsvm
from pathlib import Path
import numpy as np
alpha=3
inputmat=Extract_Pssm_hsvm.pssm_extraction(r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\positive_train_pssm_new\query_1037_pssm.txt")
S=np.asarray(inputmat).astype(np.float)
L=S.shape[0]
print(L)
S=np.transpose(S)
de1=np.zeros((20,int(alpha)))
#logic for computing de1 values
for mu in range(alpha):
    for i in range(20):
        Si_bar=np.mean(S[i])
        k=0
        # for j in range(L-mu):
            # k=k+((S[i,j]-Si_bar)*(S[i,j+mu]-Si_bar))/(L-mu)
        k=np.matmul((S[i,:L-mu]-Si_bar),(S[i,mu:L]-Si_bar))/(L-mu)
        de1[i,mu]=k
print("shape of de1",de1.shape)
alpha=3
#logic_for_de2
de2=np.zeros((20,19,alpha))
for mu in range(alpha):
    for i1 in range(20):
        f=0
        Si1_bar=np.mean(S[i1])
        for i2 in range(20):
            v=0
            Si2_bar=np.mean(S[i2])
            if(i1==i2):
                m=0
                continue
            # for j in range(L-mu):
                # v=v+((S[i1,j]-Si1_bar)*(S[i2,j+mu]-Si2_bar))/(L-mu)
            v=np.matmul((S[i1,:(L-mu)]-Si1_bar),(S[i2,mu:L]-Si2_bar))
            if(m==0):   
                de2[i1,i2-1,mu]=v
            else:
                de2[i1,i2,mu]=v

de2=np.reshape(de2,(19,20,3))
print(de2.shape)
print("the de2 is:",de2)
V_e=np.vstack([de2,[de1]])
print(V_e.shape)







