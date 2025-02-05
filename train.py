import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import test_dataset,train_dataset
from model import Model
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper params
epochs = 100
batch_size = 64
lr = 0.003
vocab_size = 30522
hidden_dim = 512
num_heads = 4
num_layers = 6
out_dim =128

"loading the model"
model = Model(vocab_size, hidden_dim, num_heads, num_layers,out_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)

loss_fn = nn.BCELoss() #deciding the loss is the important factor here 

parameters = sum(p.numel() for p in model.parameters())
print("model parameters",parameters)

"loading the data"
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size)
print("length of train datalaoder",len(train_dataloader))
print("length of test datalaoder",len(test_dataloader))

for epoch in range(epochs):
    running_loss = 0
    running_loss_test = 0
    model.train()
    for x1,x2,y in tqdm(train_dataloader):
        x1 = x1.to(device) # sentence1
        x2 = x2.to(device) # sentence2
        y = y.to(device).float()  # targets
        logits1 = model(x1)
        logits2 = model(x2)
        similarity = torch.cosine_similarity(logits1,logits2,dim=1)
        loss = loss_fn(torch.sigmoid(similarity),y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss+=loss.item()
        
    print(f"Epoch : {epoch} || Loss : {running_loss/len(train_dataloader):.4f}")
    
    model.eval()
    for x1,x2,y in tqdm(test_dataloader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device).float()
        with torch.no_grad():
          logits1 = model(x1)
          logits2 = model(x2)
        similarity = torch.cosine_similarity(logits1,logits2,dim=1)
        loss_test = loss_fn(torch.sigmoid(similarity),y)
       
        running_loss_test+=loss_test.item()
    print(f"Epoch : {epoch} || Loss_test : {running_loss_test/len(test_dataloader):.4f}")
    
    print("saving the model....")
    torch.save(model.state_dict(),f'.\checkpoints\model_{epoch}.pth')
    
      
    
    
    
        