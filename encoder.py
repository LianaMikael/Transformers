import torch
import torch.nn as nn
import copy
from attention import MultiHeadAttention
from feedforward import PositionwiseFeedforward
from embeddings import Embeddings

class Encoder(nn.Module):
  ''' Encodes a source batch using a stack of identical encoder layers ''' 
  def __init__(self, embedding_layer, encoder_layer, num_layers, dropout):
    super(Encoder, self).__init__()

    self.embed = embedding_layer
    self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
    self.dropout = nn.Dropout(dropout)

  def forward(self, source, mask):
    ''' Applies combined embeddings and applies encoder layer sequentially '''
    source = self.dropout(self.embed(source))
    for layer in self.layers:
      source = layer(source, mask)
    return source

class EncoderLayer(nn.Module):
  ''' A single encoder layer consisting of multi-head self-attention and positionwise feedforward layer'''
  def __init__(self, hidden_dim, num_heads, inner_dim, dropout):
    super(EncoderLayer, self).__init__()

    self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
    self.feedforward = PositionwiseFeedforward(hidden_dim, inner_dim, dropout)

    self.attention_norm = nn.LayerNorm(hidden_dim)
    self.feedforward_norm = nn.LayerNorm(hidden_dim)

    self.dropout = nn.Dropout(dropout)

  def forward(self, source, source_mask):
    '''
    Args:
      source: tensor [batch_size, source_len, hidden_dim]
      source_mask: tensor [batch_size, 1, 1, source_len]
    Returns:
      encdoed_source: [batch_size, source_len, hidden_dim]
    '''
    self_att_sublayer = self.self_attention(source, source, source, source_mask)

    # layernorm over residual connection of original source + dropout over self-attention sublayer
    self_att_output = self.attention_norm(source + self.dropout(self_att_sublayer))

    feedforward_sublayer = self.feedforward(self_att_output)

    # layernorm over residual connection of original source + dropout over positionwise feedforward sublayer
    encoded_source = self.feedforward_norm(self_att_output + self.dropout(feedforward_sublayer))

    return encoded_source 

if __name__ == '__main__':
    # sanity check for encoder 
    batch_size = 64
    hidden_dim = 512
    input_dim = 100
    max_length = 100
    num_heads = 8
    inner_dim = 1024
    dropout = 0.1
    num_layers = 12
    pad_id = 0
    source_len = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_layer = Embeddings(input_dim, hidden_dim, max_length, device, static=True)
    encoder_layer = EncoderLayer(hidden_dim, num_heads, inner_dim, dropout)
    encoder = Encoder(embedding_layer, encoder_layer, num_layers, dropout)

    source = torch.LongTensor(batch_size, source_len).random_(input_dim)
    source_mask = (source != pad_id).unsqueeze(1).unsqueeze(2)
    out = encoder(source, source_mask)