import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class ParaphraseDetector(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", hidden_size=768, num_heads=4, num_layers=2, dropout=0.45):
        super(ParaphraseDetector, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        # model for creating the embeddings
        
        for param in self.encoder.parameters():
            param.requires_grad = False  

        for param in self.encoder.encoder.layer[-2:].parameters():
            param.requires_grad = True
    
        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, 2) 
        
    def forward(self, input_ids, attention_mask=None):
        
        seq_length = input_ids.size(1)
        split_point = seq_length // 2
        
        input_ids1 = input_ids[:, :split_point]
        input_ids2 = input_ids[:, split_point:]
        
        if attention_mask is not None:
            attention_mask1 = attention_mask[:, :split_point]
            attention_mask2 = attention_mask[:, split_point:]
        else:
            attention_mask1 = attention_mask2 = None
        
        encoded1 = self.encoder(input_ids1, attention_mask=attention_mask1)[0]
        encoded2 = self.encoder(input_ids2, attention_mask=attention_mask2)[0]
  
        for mha in self.mha_layers:
            attn1, _ = mha(encoded1.transpose(0, 1), encoded1.transpose(0, 1), encoded1.transpose(0, 1))
            attn2, _ = mha(encoded2.transpose(0, 1), encoded2.transpose(0, 1), encoded2.transpose(0, 1))
            
            cross1, _ = mha(attn1, attn2, attn2)
            cross2, _ = mha(attn2, attn1, attn1)
      
            encoded1 = encoded1 + cross1.transpose(0, 1)
            encoded2 = encoded2 + cross2.transpose(0, 1)
        
        pooled1 = torch.mean(encoded1,dim=1)
        pooled2 = torch.mean(encoded2,dim=1)
        
        combined = torch.cat([pooled1, pooled2], dim=1)
        output = self.linear1(combined)
        output = F.relu(output)
        output = self.dropout(output)
        logits = self.linear2(output)
        
        return logits, encoded1, encoded2