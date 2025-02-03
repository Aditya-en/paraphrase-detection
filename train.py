import torch
import torch.nn as nn
import torch.optim as optim
from data import train_dataloader,test_dataloader

from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper params
epochs = 100
batch_size = 64
lr = 0.001
vocab_size = 1000
hidden_dim = 64
num_heads = 4
num_layers = 4
model = Model(vocab_size, hidden_dim, num_heads, num_layers)

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss()

for epoch in range(epochs):
    running_loss = 0
    running_loss_test = 0
    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        logits1 = model(x[0])
        logits2 = model(x[1])
        similarity = torch.tensordot(logits1,logits2,dims=([1,1],))
        loss = loss_fn(similarity,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss+=loss.item()
        
    print(f"Epoch : {epoch} || Loss : {running_loss/len(train_dataloader):.4f}")
    
    model.eval()
    for x,y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
          logits1 = model(x[0])
          logits2 = model(x[1])
        similarity = torch.tensordot(logits1,logits2,dims=([1,1],))
        loss_test = loss_fn(similarity,y)
       
        running_loss_test+=loss_test.item()
    print(f"Epoch : {epoch} || Loss_test : {running_loss_test/len(test_dataloader):.4f}")
    
    print("saving the model....")
    torch.save(model.state_dict(),f'.\checkpoints\model_{epoch}.pth')
    
      
    
    
    
        