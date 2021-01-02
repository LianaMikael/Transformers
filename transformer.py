import torch
import torch.nn as nn

class Transformer(nn.Module):
  ''' Transformer model incorporating Encoder and Decoder and creation of masks '''
  def __init__(self, encoder, decoder, pad_id, device):
    super(Transformer, self).__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.pad_id = pad_id 
    self.device = device 

  def forward(self, source, target):
    encoder_out, source_mask = self.encode(source)
    return self.decode(target, encoder_out, source_mask)

  def encode(self, source):
    ''' Generates the source mask and applies Encoder network '''
    source_maks = (source != self.pad_id).unsqueeze(1).unsqueeze(2)
    return self.encoder(source, source_maks), source_maks

  def decode(self, target, enc_source, source_mask):
    ''' Generates the target mask and applies Decoder network '''
    target_pad_mask = (target != self.pad_id).unsqueeze(1).unsqueeze(2)
    target_len = target.shape[1]
    # to ensure that future positions are not attended to 
    target_subsequent_mask = torch.tril(torch.ones((target_len, target_len), device=self.device)).bool()
    combined_target_maks = target_pad_mask & target_subsequent_mask
    return self.decoder(target, combined_target_maks, enc_source, source_mask)

  def save(self, path):
    torch.save(self.state_dict(), path)
    print('Saved model at', path)

  def load(self, path):
    self.load_state_dict(torch.load(path))
    print('Loaded model from', path)