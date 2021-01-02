import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    ''' Multi-head attention layer '''
    def __init__(self, hidden_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.calc_head_dim()
        
        # linear layers for key, query and value 
        self.linear_query = nn.Linear(hidden_dim, hidden_dim)
        self.linear_key = nn.Linear(hidden_dim, hidden_dim)
        self.linear_value = nn.Linear(hidden_dim, hidden_dim)
        
        # linear layer for the output
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, key, query, value, mask=None):
        '''
        Args:
            key: tensor [batch_size, key_len, hidden_dim]
            query: tesnor [batch_size, query_len, hidden_dim]
            value: tensor [batch_size, value_len, hidden_dim]
        Returns:
            Attention output projection: tensor [batch_size, query_len, hidden_dim]
        '''
        
        # apply linear projections through key, query and value
        # dimensions remain unchanged 
        query_proj = self.linear_query(query)
        key_proj = self.linear_key(key)
        value_proj = self.linear_value(value)
        
        # transpose to [batch_size, num_heads, len, head_dim] to apply attention layers to each head 
        batch_size = key.shape[0]
        query_proj = query_proj.view(batch_size, query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key_proj = key_proj.view(batch_size, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        value_proj = value_proj.view(batch_size, value.shape[1], self.num_heads,self.head_dim).transpose(1, 2)
        
        att_out = self.attention(query_proj, key_proj, value_proj, True, mask, self.dropout)

        att_out = att_out.permute(0, 2, 1, 3).contiguous()
        att_out = att_out.view(att_out.shape[0], -1, self.hidden_dim) 

        att_proj = self.linear_out(att_out) 

        return att_proj
        
    def attention(self, query, key, value, scale=None, mask=None, dropout=None):
        ''' Calculates dot-product attention '''
        att_scores = torch.matmul(query, key.transpose(-2, -1)) 
        
        if scale is not None:
          att_scores /= math.sqrt(query.shape[-1])
        
        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, -1e9)
        
        probs = F.softmax(att_scores, dim = -1)
        
        if dropout is not None:
            probs = self.dropout(probs)
            
        att_out = torch.matmul(probs, value) # [batch_size, num_heads, query_len, head_dim]

        return att_out

    def calc_head_dim(self):
      if self.hidden_dim % self.num_heads != 0:
        raise ValueError('Hidden dimension must be divisible by number of heads to calculate each head dimension.')
      return self.hidden_dim // self.num_heads

if __name__ == '__main__':
    # sanity check for the multi-head attention module
    batch_size = 64
    hidden_dim = 512
    num_heads = 4
    dropout = 0.1
    seq_len = 100

    key = torch.rand(batch_size, seq_len, hidden_dim)
    query = torch.rand(batch_size, seq_len, hidden_dim)
    value = torch.rand(batch_size, seq_len, hidden_dim)

    att_module = MultiHeadAttention(hidden_dim, num_heads, dropout)
    att_out = att_module(query, key, value)

    assert att_out.shape == key.shape 