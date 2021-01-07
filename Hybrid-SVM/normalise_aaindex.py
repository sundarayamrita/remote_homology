from pathlib import Path
import numpy as np

def normalise(filepath):

	with open(filepath, 'r') as file:
		aaindex = file.readlines()

	I_hat = [line.strip().split() for line in aaindex if '>' not in line]
#	for line in aaindex:
#		if '>' not in line:
#			I_hat.append(line.strip().split())

	I_hat = np.asarray(I_hat).astype(np.float32)
#	print(I_hat.shape)
	vector_shape = (I_hat.shape[0],1)

	I_hat_avg = np.mean(I_hat, 1).reshape(vector_shape)

	rms = np.sqrt(np.mean(np.square(I_hat - I_hat_avg), 1)).reshape(vector_shape)
#	print(rms.shape)

	I = (I_hat - I_hat_avg) / rms
	return I

if __name__ == "__main__":
	file = Path.cwd() / "aaindex_format.txt"
	normalise_aaindex(file)
