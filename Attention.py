import torch
from torch import float32, nn
from torch.nn.parameter import Parameter
from math import sqrt

class Attention(nn.Module):
    def __init__(self, attender_dim, attendee_dim, output_dim, kq_dim) -> None:
        super().__init__()
        self.W_key = Parameter(torch.randint(-1, 1, (attender_dim, kq_dim), dtype=float32))
        self.W_query = Parameter(torch.randint(-1, 1, (attendee_dim, kq_dim), dtype=float32))
        self.W_value = Parameter(torch.randint(-1, 1, (attendee_dim, output_dim), dtype=float32))
        self.b = Parameter(torch.zeros(output_dim))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, attenders, attendees):
        # calculate key, query, and value matrices
        k = attendees @ self.W_key
        q = attenders @ self.W_query
        v = attenders @ self.W_value + self.b
        # calculate attention weights
        y = (q @ k.T) / sqrt(self.W_key.size(dim=1))
        y = self.softmax(y)
        y = y @ v
        return y

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, kq_dim) -> None:
        super().__init__()
        self.attention = Attention(input_dim, input_dim, output_dim, kq_dim)
    
    def forward(self, X):
        return self.attention(X, X)
     