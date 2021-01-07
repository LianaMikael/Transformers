import torch
import torch.nn as nn
import copy
from attention import MultiHeadAttention
from feedforward import PositionwiseFeedforward
from embeddings import Embeddings

class Decoder(nn.Module):
  ''' Decodes a target batch using a stack of identical decoder layers '''
  def __init__(self, output_dim, hidden_dim, embedding_layer, decoder_layer, num_layers, dropout):
    super(Decoder, self).__init__()

    self.embed = embedding_layer
    self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(hidden_dim, output_dim)
    self.log_softmax = nn.LogSoftmax(dim=1)

  def forward(self, target, target_mask, encoded_source, source_mask):
    ''' Applies combined embeddings, decoder layers, linear and softmax layers'''
    target = self.dropout(self.embed(target))
    
    for layer in self.layers:
      target = layer(target, target_mask, encoded_source, source_mask)
    
    target_proj = self.linear(target)
    return self.log_softmax(target_proj)

class DecoderLayer(nn.Module):
  ''' Decoder layer consisting of three sequential components: 
      masked multi-head self-attention, 
      encoder-decoder multi-head attention,
      positionwise feedforward layer'''

  def __init__(self, hidden_dim, num_heads, inner_dim, dropout):
    super(DecoderLayer, self).__init__()

    self.masked_self_att = MultiHeadAttention(hidden_dim, num_heads, dropout)
    self.combined_att = MultiHeadAttention(hidden_dim, num_heads, dropout)
    self.feedforward = PositionwiseFeedforward(hidden_dim, inner_dim, dropout)

    self.masked_self_att_norm = nn.LayerNorm(hidden_dim)
    self.combined_att_norm = nn.LayerNorm(hidden_dim)
    self.feedforward_norm = nn.LayerNorm(hidden_dim)

    self.dropout = nn.Dropout(dropout)

  def forward(self, target, target_mask, encoded_source, source_mask):
    '''
    Args:
      target: tensor [batch_size, target_len, hidden_dim]
      target_mask: tensor [batch_size, 1, target_len, target_len]
      encoded_source: tensor [batch_size, source_len, hidden_dim]
      source_mask: tensor [batch_szie, 1, 1, source_len]
    Returns:
      target_output: tensor [batch_size, target_len, hidden_dim]
    '''

    target_self_att = self.masked_self_att(target, target, target, target_mask)
    new_target = self.masked_self_att_norm(target + self.dropout(target_self_att))

    combined_att = self.combined_att(encoded_source, new_target, encoded_source, source_mask)
    new_target = self.combined_att_norm(new_target + self.dropout(combined_att))

    target_feedforward = self.feedforward(new_target)
    target_output = self.feedforward_norm(new_target + self.dropout(target_feedforward))

    return target_output

if __name__ == '__main__':
    # sanity check for decoder 
    batch_size = 64
    hidden_dim = 512
    input_dim = 100
    output_dim = 100
    max_length = 100
    num_heads = 8
    inner_dim = 1024
    dropout = 0.1
    num_layers = 12
    pad_id = 0
    seq_len = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_layer = Embeddings(input_dim, hidden_dim, max_length, device, static=False)
    decoder_layer = DecoderLayer(hidden_dim, num_heads, inner_dim, dropout)
    decoder = Decoder(output_dim, hidden_dim, embedding_layer, decoder_layer, num_layers, dropout)

    source = torch.LongTensor(batch_size, seq_len).random_(input_dim)
    source_mask = (source != pad_id).unsqueeze(1).unsqueeze(2)

    target = torch.LongTensor(batch_size, seq_len).random_(input_dim)
    target_mask = (target != pad_id).unsqueeze(1).unsqueeze(2)

    encoded_source = torch.rand(batch_size, seq_len, hidden_dim)
    output = decoder(target, target_mask, encoded_source, source_mask)