from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import gc
import matplotlib.pyplot as plt
from data import get_dataloaders
from model_final import ParaphraseDetector

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs=5, patience=3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * num_training_steps,
        num_training_steps=num_training_steps
    )
    
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []
        
        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs,_,_ = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_losses.append(total_loss / len(train_dataloader))
        
        train_f1 = f1_score(train_labels, train_preds, average='binary')
        train_acc = accuracy_score(train_labels, train_preds)
        
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs,_,_ = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_losses.append(val_loss / len(val_dataloader))
        
        val_f1 = f1_score(val_labels, val_preds, average='binary')
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_losses[-1]:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_losses[-1]:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}\n')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
        
        torch.cuda.empty_cache()
        gc.collect()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_plot.png')
    plt.show()
    
    return model


#hyperparameters
num_epochs = 2
lr = 1e-5
weight_decay = 0.01
batch_size = 16
num_layers = 2
num_heads = 4
dropout = 0.45

train_loader, val_loader = get_dataloaders(batch_size=batch_size)

model = ParaphraseDetector(num_heads=num_heads,num_layers=num_layers,dropout=dropout)
model_parameters = sum([p.numel() for p in model.parameters()])
print("Model parameters", model_parameters)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
model = train_model(model, train_loader, val_loader, optimizer, num_epochs=num_epochs)

print("Training success.. saving the model...")
torch.save(model.state_dict(), 'model_final.pth')
