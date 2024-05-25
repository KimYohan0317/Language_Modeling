# import some packages you need here
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Shakespeare(Dataset):

    def __init__(self, input_file):

        # write your codes here
        with open(input_file, 'rb') as f:
            self.total_data = ''.join([
                line.strip().lower().decode('ascii', 'ignore')
                for line in f if line.strip()
            ])
        unique_chars = sorted(set(self.total_data))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}    
        
        self.seq_length = 30
        self.total_len = len(self.total_data) - self.seq_length
    def __len__(self):

        # write your codes here
        return self.total_len
    def __getitem__(self, idx):

        # write your codes here
        X_sample = self.total_data[idx:idx + self.seq_length]
        y_sample = self.total_data[idx + 1:idx + self.seq_length + 1]
        
        X_encoded = [self.char_to_idx[char] for char in X_sample]
        y_encoded = [self.char_to_idx[char] for char in y_sample]
        
        X_tensor = torch.LongTensor(X_encoded)
        
        input = F.one_hot(X_tensor, num_classes=len(self.char_to_idx)).float()
        target = torch.LongTensor(y_encoded)
        return input, target

if __name__ == '__main__':

    # write test codes to verify your implementations
    input_file = './shakespeare_train.txt'
    dataset = Shakespeare(input_file)
    first_input, first_target = dataset[0]
    print('datsaet_length: ', len(dataset))
    print('first_input: ', first_input)
    print('first_target: ', first_target)