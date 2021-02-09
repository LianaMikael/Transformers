import torch 
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
import json 
from jiwer import wer 
from transformers import BertTokenizer

class SentenceDataset(Dataset):
    ''' Dataset of source and target tensors '''
    def __init__(self, data_path, encoding_type, load_vocab=None, save_vocab=None, filter_threshold=None, num_tokens=100):
        self.data_path = data_path
        self.encoding_type = encoding_type
        self.filter_threshold = filter_threshold
        self.source_sentences, self.target_sentences = self.read_and_tokenize()
        self.num_tokens = num_tokens

        if load_vocab is not None:
            self.vocab = Vocabulary.load(load_vocab)
        else:
            self.vocab = Vocabulary.build(self.source_sentences, self.target_sentences)
            if save_vocab is not None:
                self.vocab.save(save_vocab)

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        source = self.convert_to_tensor(self.source_sentences[idx], self.vocab.source_vocab)
        target = self.convert_to_tensor(self.target_sentences[idx], self.vocab.target_vocab)
        return source, target

    def read_and_tokenize(self):
        ''' Reads and tokenizes source and target sentences '''
        source_sentences = []
        target_sentences = []

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        with open(self.data_path, 'r+') as f:
            for line in f:
                line = line.split(',')
                if self.encoding_type == 'word':
                    source_tokens = word_tokenize(line[0])
                    target_tokens = word_tokenize(line[1])
                elif self.encoding_type == 'char':
                    source_tokens = self.char_tokenize(line[0])
                    target_tokens = self.char_tokenize(line[1])
                elif self.encoding_type == 'bpe':
                    source_tokens = tokenizer.tokenize(line[0])
                    target_tokens = tokenizer.tokenize(line[1])
                    # TODO: add a custom BPE tokenizer 
                else:
                    raise NotImplementedError('Dataset only supports character-based (char), word-based (word) or bpe encoding ')

                if self.filter_threshold:
                    if wer(target_tokens, source_tokens) > self.filter_threshold:
                        continue 

                source_sentences.append(source_tokens)
                target_sentences.append(['<s>'] + target_tokens + ['</s>']) 

        return source_sentences, target_sentences

    def convert_to_tensor(self, sentence, tokens_vocab):
        ''' Converts a list of tokens into padded tokens using the corresponding vocabulary '''

        word_ids = [tokens_vocab.get(word, tokens_vocab['<unk>']) for word in sentence]
        padded_ids = self.pad_sentences(word_ids)
        return torch.tensor(padded_ids, dtype=torch.long)

    def pad_sentences(self, sentence):
        ''' Pads a sentence to a specific number of tokens '''
        assert len(sentence) > 0
        if len(sentence) < self.num_tokens:
            return sentence + [0] * (self.num_tokens - len(sentence))
        return sentence[:self.num_tokens]

    @staticmethod
    def char_tokenize(text):
        return [char for char in text.strip().replace(' ', '_')]

class Vocabulary:
    ''' Constructs a vocabulary using character-based or word-based encoding '''
    def __init__(self, source_vocab, target_vocab):

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def save(self, vocab_file):
        vocab = {'source_data': self.source_vocab, 'target_data': self.target_vocab}
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f, indent = 2)
        print('Vocabulary saved into', vocab_file)

    @staticmethod
    def build(source_sentences, target_sentences):

        assert len(source_sentences) == len(target_sentences)

        source_vocab = Vocabulary.construct_vocab(source_sentences)
        target_vocab = Vocabulary.construct_vocab(target_sentences)

        return Vocabulary(source_vocab, target_vocab)

    @staticmethod
    def construct_vocab(corpus):
        ''' Constructs a vocabulary dictionary with values corresponding to word or char ids '''
        vocab = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
        for sent in corpus:
            for token in sent:
                if token not in vocab:
                    vocab[token] = len(vocab)

        return vocab

    @staticmethod
    def load(vocab_file):
        with open(vocab_file, 'r+') as f:
            word_index = json.load(f)
            source_vocab = word_index['source_data']
            target_vocab = word_index['target_data'] 

        print('Vocabulary loaded from',  vocab_file)
        return Vocabulary(source_vocab, target_vocab)

if __name__ == '__main__':
    train_data = SentenceDataset('github_train.txt', encoding_type='char', save_vocab='vocab.json')
    print('{} training examples loaded'.format(len(train_data)))
    print('{} vocabulary items saved'.format(len(train_data.vocab.source_vocab)))
    val_data = SentenceDataset('github_val.txt', encoding_type='char', load_vocab='vocab.json')
    print('{} validation examples loaded'.format(len(val_data)))
