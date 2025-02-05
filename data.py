import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
filepath_train = 'data/labeled_final_train.csv'
filepath_test = 'data/labeled_final_test.csv'

class ParaphraseDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.data = pd.read_csv(filepath)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        encoding1 = self.tokenizer(
            row["sentence1"], padding="max_length", truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        )
        encoding2 = self.tokenizer(
            row["sentence2"], padding="max_length", truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        )

        input_ids1 = encoding1["input_ids"].squeeze(0)  
        input_ids2 = encoding2["input_ids"].squeeze(0)  
        label = torch.tensor(row["label"], dtype=torch.long)

        return (input_ids1, input_ids2, label)
      
train_dataset = ParaphraseDataset(filepath=filepath_train, tokenizer=tokenizer)
test_dataset = ParaphraseDataset(filepath=filepath_test, tokenizer=tokenizer)


