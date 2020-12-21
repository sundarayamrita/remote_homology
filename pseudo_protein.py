import numpy as np
import psuedo_protein_generation

aa = ['A','R', 'N', 'D', 'C', 'Q','E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T','W', 'Y', 'V']

pssm = np.asarray(psuedo_protein_generation.pssm_extraction())
indices = []
for mul_ali in pssm:
	indices.append(np.argmax(mul_ali, axis = 0))
indices = np.asarray(indices)
pseudo_protein = []
for idx in indices:
	pseudo_protein.append(aa[idx])
pseudo_protein = "".join(pseudo_protein)
print(pseudo_protein)
