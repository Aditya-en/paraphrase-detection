import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
filepath_train = 'data\msr_paraphrase_train.txt'
filepath_test = 'data\msr_paraphrase_test.txt'

class ParaphraseDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=256): 
        self.data = pd.read_csv(filepath, sep='\t', on_bad_lines='skip')
        self.data = self.data.rename(columns={'Quality': 'label', '#1 String': 'sentence1', '#2 String': 'sentence2'})
        self.data = self.data.dropna(subset=["sentence1", "sentence2"])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        encoding = self.tokenizer(
            row["sentence1"],
            row["sentence2"],
            padding="max_length",
            truncation="longest_first", 
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(row["label"], dtype=torch.long)
        }

def get_dataloaders(train_filepath=filepath_train, val_filepath=filepath_test, batch_size=16):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    train_dataset = ParaphraseDataset(filepath=train_filepath, tokenizer=tokenizer)
    val_dataset = ParaphraseDataset(filepath=val_filepath, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader

