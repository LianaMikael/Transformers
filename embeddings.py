import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
  ''' Implements sum of token embeddings and positional embeddings (learned or static) '''
  def __init__(self, input_dim, hidden_dim, max_length, device, static=False):
    super(Embeddings, self).__init__()

    self.token_embeddings = nn.Embedding(input_dim, hidden_dim)
    self.max_length = max_length
    self.hidden_dim = hidden_dim
    self.device = device
    self.static = static

    if static:
      self.pos_embeddings = self.static_encoding()
    else:
      self.pos_embeddings = nn.Embedding(max_length, hidden_dim)

  def forward(self, source):
    token_embed = self.token_embeddings(source) * math.sqrt(self.hidden_dim)
    if self.static:
      return token_embed + self.pos_embeddings[:, :token_embed.size(1)]

    positions = torch.arange(source.shape[1]).unsqueeze(0).repeat(source.shape[0], 1).to(self.device)    
    return token_embed + self.pos_embeddings(positions)

  def static_encoding(self): 
    ''' Static positional encoding from http://nlp.seas.harvard.edu/2018/04/03/attention.html#embeddings-and-softmax'''
    pos_enc = torch.zeros(self.max_length, self.hidden_dim)
    positions = torch.arange(self.max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.hidden_dim, 2) * -(math.log(10000.0) / self.hidden_dim))
    pos_enc[:, 0::2] = torch.sin(positions * div_term)
    pos_enc[:, 1::2] = torch.cos(positions * div_term)
    return pos_enc.unsqueeze(0).to(self.device)