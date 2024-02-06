import torch
from torch.utils.data import Dataset
import numpy as np

class TokenDataset(Dataset):
    def __init__(self, file_path, seq_length):
        self.file_path = file_path
        self.seq_length = seq_length
        self.data = np.memmap(file_path, dtype=np.int32, mode='r')
        self.num_samples = len(self.data) // seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_index = idx * self.seq_length
        end_index = start_index + self.seq_length + 1
        sequence = self.data[start_index:end_index]
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target_seq = torch.tensor(sequence[1:], dtype=torch.long)
        return input_seq, target_seq
