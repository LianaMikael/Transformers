import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataset import SentenceDataset
from embeddings import Embeddings
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from transformer import Transformer
from infer import Inferencer

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-train_file', type=str, default='github_train.txt', help='path to training dataset file')
parser.add_argument('-val_file', type=str, default='github_val.txt', help='path to validation dataset file')
parser.add_argument('-encoding_type', type=str, default='char', help='char or word')
parser.add_argument('-save_vocab', type=str, default='vocab.json', help='json file to save vocabulary into')
parser.add_argument('-load_vocab', type=str, default=None, help='json file to load vocabulary from')
parser.add_argument('-embedding_type', type=str, default='learned', help='learned or static')
parser.add_argument('-batch_size', type=int, default=128, help='batch size of training')
parser.add_argument('-filter_threshold', type=float, default=0.2, help='threshold to filter out data according to WER or CER')
parser.add_argument('-hidden_dim', type=int, default=256, help='dimentionality of the model')
parser.add_argument('-inner_dim', type=int, default=512, help='dimentionality to upsamle the model in position-wise feedforward layer')
parser.add_argument('-num_enc_layers', type=int, default=3, help='number of encoder layers')
parser.add_argument('-num_dec_layers', type=int, default=3, help='number of decoder layers')
parser.add_argument('-num_enc_heads', type=int, default=8, help='number of multi-head attention heads in encoder')
parser.add_argument('-num_dec_heads', type=int, default=8, help='number of multi-head attention heads in decoder')
parser.add_argument('-max_len', type=int, default=100, help='maximum number of input tokens')
parser.add_argument('-dropout', type=float, default=0.1, help='dropout')
parser.add_argument('-lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('-epochs', type=int, default=1, help='number of epochs')
parser.add_argument('-save_every', type=int, default=50, help='number of iteraitions to save and evaluate')
parser.add_argument('-save_model', type=str, default='best_model.bin', help='path to save model checkpoint')
parser.add_argument('-load_model', type=str, default=None, help='path to model checkpoint to load')
parser.add_argument('-mode', type=str, default='train', help='train or test mode')
args = parser.parse_args()

def train(model, train_loader, val_loader, optimizer, loss_function, device):
    writer = SummaryWriter()
    model.train()
    val_losses = []
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            source = batch[0].to(device)
            target = batch[1].to(device)

            output = model(source, target[:,:-1])

            output = output.contiguous().view(-1, output.shape[-1])
            target = target[:,1:].contiguous().view(-1)

            loss = loss_function(output, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            optimizer.zero_grad()
            
            if i % args.save_every == 0:
                val_loss = evaluate(model, val_loader, loss_function, device)

                writer.add_scalar('Loss/train', loss.item(), epoch*len(train_loader)+i)
                writer.add_scalar('Loss/val', val_loss, epoch*len(train_loader)+i)
                
                model.train()
                print('epoch: {}, train iteration: {}, train loss {}, val loss {}'.format(epoch, epoch*len(train_loader)+i, loss.item(), val_loss))
                val_losses.append(val_loss)

                if val_loss <= min(val_losses):
                    model.save(args.save_model)
    writer.close()

def evaluate(model, val_loader, loss_function, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            source = batch[0].to(device)
            target = batch[0].to(device)

            output = model(source, target[:,:-1])

            output = output.contiguous().view(-1, output.shape[-1])
            target = target[:,1:].contiguous().view(-1)

            loss = loss_function(output, target)
            val_loss += loss.item()
            return val_loss
    return val_loss / len(val_loader)

def count_parames(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)

def main():
    train_data = SentenceDataset(args.train_file, encoding_type=args.encoding_type, filter_threshold=args.filter_threshold)
    val_data = SentenceDataset(args.val_file, encoding_type=args.encoding_type, filter_threshold=args.filter_threshold)

    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, args.batch_size)

    print(len(train_loader))

    input_dim = len(train_data.vocab.source_vocab)
    output_dim = len(train_data.vocab.target_vocab)
    static = args.embedding_type == 'static'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc_embedding = Embeddings(input_dim, args.hidden_dim, args.max_len, device, static)
    encoder_layer = EncoderLayer(args.hidden_dim, args.num_enc_heads, args.inner_dim, args.dropout)
    encoder = Encoder(enc_embedding, encoder_layer, args.num_enc_layers, args.dropout)

    dec_embedding = Embeddings(input_dim, args.hidden_dim, args.max_len, device, static)
    decoder_layer = DecoderLayer(args.hidden_dim, args.num_dec_heads, args.inner_dim, args.dropout)
    decoder = Decoder(output_dim, args.hidden_dim, dec_embedding, decoder_layer, args.num_dec_layers, args.dropout)

    pad_id = train_data.vocab.source_vocab['<pad>']

    model = Transformer(encoder, decoder, pad_id, device)

    print('Transformer has {:,} trainable parameters'.format(count_parames(model)))

    if args.load_model is not None:
        model.load(args.load_model)
    else:
        model.apply(init_weights)

    if args.mode == 'test':
        inferencer = Inferencer(model, train_data.vocab, device)
        greedy_out = inferencer.infer_greedy('helo world, I m testin a typo corector')
        print(greedy_out)

    elif args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

        loss_function = nn.NLLLoss(ignore_index=pad_id)

        print('Started training...')
        train(model, train_loader, val_loader, optimizer, loss_function, device)
    
    else:
        raise ValueError('Mode not recognized')

if __name__ == '__main__':
    main()