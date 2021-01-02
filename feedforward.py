import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedforward(nn.Module):
  ''' Feedforward layer with two linear transformations and ReLU in between'''
  def __init__(self, hidden_dim, inner_dim, dropout):
    super(PositionwiseFeedforward, self).__init__()

    self.linear_1 = nn.Linear(hidden_dim, inner_dim)
    self.linear_2 = nn.Linear(inner_dim, hidden_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, encoded_input):
    ''' encoded_input: [batch_size, seq_len, hidden_dim] '''
    return self.linear_2(self.dropout(F.gelu(self.linear_1(encoded_input)))) 

if __name__ == '__main__':
    # sanity check for PositionwiseFeedforward sub-layer 
    batch_size = 64
    hidden_dim = 512
    inner_dim = 1024
    dropout = 0.1
    seq_len = 100

    x = torch.rand(batch_size, seq_len, hidden_dim)
    positionwise_feeedforward = PositionwiseFeedforward(hidden_dim, inner_dim, dropout)
    output = positionwise_feeedforward(x)

    assert output.shape == x.shape