import numpy as np
from Bio import SeqIO
import Parameters

class prepareData():
    def __init__(self):
        self.DATA_HOME = 'data/'

        self.window_size = Parameters.window_size
        self.sequenceLength = Parameters.sequence_length
        self.coding = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F',
        'P', 'S', 'T', 'W', 'Y', 'V']
        self.aaVal = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


    def read_fasta(self, path):

        seq_record = list(SeqIO.parse(path, 'fasta'))
        coding_record = []
        protein_names = []
        for record in seq_record:
            sequence = record.seq
            coding = []
            protein_names.append(record.id)
            for aa in sequence:
                aa = aa.upper()
                if aa not in self.coding:
                    continue
                coding.append(self.coding.index(aa))
            coding_record.append(coding)
        return coding_record, protein_names



    def genertateMat(self, path,npy_path = None):
        coding_record, protein_names  = self.read_fasta(path)
        wside = (self.window_size - 1)/2
        mats = np.load(npy_path)
        # mats = np.asarray(
        #     np.zeros([len(coding_record), self.sequenceLength, 20*self.window_size]))
        # index = 0

        # for seq in coding_record:

        #     seqLength = len(seq)
        #     if seqLength > self.sequenceLength:
        #         seqLength = self.sequenceLength
        #     # else:
        #     #     x = 'shorter than ' + str(seqLength)

        #     inputVector= np.zeros([self.sequenceLength, len(self.coding)*self.window_size])
        #     for i in range(seqLength):
        #         starti = i - wside
        #         for j in range(self.window_size):
        #             iter = starti + j
        #             if (iter < 0 or iter >= seqLength):
        #                 continue
        #             iter=int(iter)
        #             aaCharPos = seq[iter]
        #             inputVector[i][j*len(self.coding)+aaCharPos] = self.aaVal[aaCharPos]

        #     mats[index] = inputVector
        #     index = index + 1


        return mats, protein_names










    def generateInputData(self, args):
        pos_train_mat, pos_train_names = self.genertateMat(args.pos_train_dir,r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\cnn_classifier\indexed_files\a.7.1\pos-train.a.7.1_corr.npy")
        neg_train_mat, neg_train_names = self.genertateMat(args.neg_train_dir,r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\cnn_classifier\indexed_files\a.7.1\neg-train.a.7.1_corr.npy")
        pos_test_mat, pos_test_names = self.genertateMat(args.pos_test_dir,r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\cnn_classifier\indexed_files\a.7.1\pos-test.a.7.1_corr.npy")
        neg_test_mat, neg_test_names = self.genertateMat(args.neg_test_dir,r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\cnn_classifier\indexed_files\a.7.1\neg-test.a.7.1_corr.npy")
        pos_train_num = pos_train_mat.shape[0]
        print("shape: pos_train_num",pos_train_num)
        neg_train_num = neg_train_mat.shape[0]
        print("shape: neg_train_num",neg_train_num)
        pos_test_num = pos_test_mat.shape[0]
        print("shape: pos_test_num",pos_test_num)
        neg_test_num = neg_test_mat.shape[0]
        print("shape: neg_test_num",neg_test_num)
        train_mat = np.vstack((pos_train_mat, neg_train_mat))
        test_mat = np.vstack((pos_test_mat, neg_test_mat))
        train_label = np.hstack((np.ones(pos_train_num), np.zeros(neg_train_num)))
        test_label = np.hstack((np.ones(pos_test_num), np.zeros(neg_test_num)))
        pos_test_names.extend(neg_test_names)
        test_names = pos_test_names
        return (train_mat, train_label, test_mat, test_label, test_names)


    def generateTestingSamples(self, args):

        pos_test_mat, pos_test_names = self.genertateMat(args.pos_test_dir,r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\cnn_classifier\indexed_files\c.1.4\pos-test.c.1.4_corr.npy")

        neg_test_mat, neg_test_names = self.genertateMat(args.neg_test_dir,r"C:\Users\meera\OneDrive\Desktop\Homology\blast_gen_files_single\Remote-homology\cnn_classifier\indexed_files\c.1.4\neg-test.c.1.4_corr.npy")


        pos_test_num = pos_test_mat.shape[0]
        neg_test_num = neg_test_mat.shape[0]

        test_mat = np.vstack((pos_test_mat, neg_test_mat))

        test_label = np.hstack((np.ones(pos_test_num), np.zeros(neg_test_num)))
        pos_test_names.extend(neg_test_names)
        test_names = pos_test_names
        return (test_mat, test_label, test_names)





