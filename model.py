import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim//num_heads
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.wo = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        b, s, d = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_heads, self.head_dim)
        v = v.view(b, s, self.num_heads, self.head_dim)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        atten_score = torch.softmax((q@k.transpose(2,3))/(self.head_dim**0.5), dim=-1)
        o = atten_score@v
        o = o.transpose(1,2).contiguous().view(b, s, d)
        return self.wo(o)
        
class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.mha = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = self.layernorm1(x)
        x = self.mha(x)
        x = self.ffn(x)
        x = self.layernorm2(x)
        return x
        
class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_heads,
        num_layers,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size,hidden_dim)
        self.layers = nn.ModuleList([EncoderBlock(self.hidden_dim, self.num_heads) for _ in range(num_layers)])
        # self.out = nn.Linear(self.hidden_dim, self.vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = torch.mean(x,dim=1)
        # x = self.out(x)
        return x
        
