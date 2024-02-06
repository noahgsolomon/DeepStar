import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-6) + self.beta

class DecoderTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoding = tiktoken.get_encoding("r50k_base")
        self.emb_size = self.encoding.n_vocab
        self.emb_channels = 128
        self.emb = nn.Embedding(self.emb_size, self.emb_channels)
        self.qW = nn.Parameter(torch.randn(self.emb_channels, self.emb_channels))
        self.kW = nn.Parameter(torch.randn(self.emb_channels, self.emb_channels))
        self.vW = nn.Parameter(torch.randn(self.emb_channels, self.emb_channels))
        self.oW = nn.Parameter(torch.randn(self.emb_channels, self.emb_channels))
        self.gamma = nn.Parameter(torch.ones(self.emb_channels))
        self.beta = nn.Parameter(torch.zeros(self.emb_channels))
        self.num_heads = 8
        self.head_dim = self.emb_channels // self.num_heads
        assert self.head_dim * self.num_heads == self.emb_channels, "emb_channels must be divisible by num_heads"
        self.l1 = nn.Linear(self.emb_channels, 250)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(250, self.emb_channels)
        self.ln1 = LayerNorm(self.emb_channels)
        self.ln2 = LayerNorm(self.emb_channels)
        self.linear = nn.Linear(self.emb_channels, self.emb_size)
        self.n = 3


    def positional_encoding(self, x):
        _, seq_length, d = x.shape
        encoding = x.clone()
        pos = torch.arange(seq_length).unsqueeze(1)
        i = torch.arange(d).unsqueeze(0)
        factor = 10000 ** (2 * i / d)
        position_tensor = pos / factor
        for i in i[0]:
            encoding += torch.sin(position_tensor) if i % 2 == 0 else torch.cos(position_tensor)
        return encoding
    
    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def self_attention(self, x):
        batch_size = x.shape[0]
        q = self.split_heads(x @ self.qW, batch_size)
        k = self.split_heads(x @ self.kW, batch_size)
        v = self.split_heads(x @ self.vW, batch_size)
        qK = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        qK = self.mask(qK)
        attention_weights = F.softmax(qK, dim=-1)
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_channels)
        output = output @ self.oW
        output += x
        return output
        
    def mask(self, x):
        seq_length = x.shape[2]
        mask = torch.tril(torch.ones((seq_length, seq_length), device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(1)
        mask = mask.repeat(x.shape[0], self.num_heads, 1, 1)
        return x.masked_fill(mask == 0, float('-inf'))
    
    def feed_forward(self, x):
        y = self.l1(x)
        y = self.relu(y)
        y = self.l2(y)
        y += x
        return y

    def forward(self, x):
        max_length = max(t.size(0) for t in x)
        padded = [F.pad(t, (0, max_length - t.size(0))) for t in x]
        input_tensor = torch.stack(padded)
        x = self.emb(input_tensor)
        x = self.positional_encoding(x)
        for _ in range(self.n):
            x = self.self_attention(x)
            x = self.ln1(x)
            x = self.feed_forward(x)
            x = self.ln2(x)
        x = self.linear(x)
        return x