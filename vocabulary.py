from nltk.tokenize import word_tokenize
import json 

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

    @classmethod
    def read_and_tokenize(cls, data_path, encoding_type=None):
        ''' Reads and tokenizes source and target sentences '''
        source_sentences = []
        target_sentences = []
        with open(data_path, 'r+') as f:
            for line in f:
                line = line.split(',')
                if encoding_type == 'word':
                    source_tokens = word_tokenize(line[0])
                    target_tokens = word_tokenize(line[1])
                elif encoding_type == 'char':
                    source_tokens = cls.char_tokenize(line[0])
                    target_tokens = cls.char_tokenize(line[1])
                else:
                    raise NotImplementedError('Vocabulary only supports character-based (char) or word-based (word) encoding')

                source_sentences.append(source_tokens)
                target_sentences.append(['<s>'] + target_tokens + ['</s>']) 

        return source_sentences, target_sentences

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
    def char_tokenize(text):
        return [char for char in text.strip().replace(' ', '_')]

    @staticmethod
    def load(vocab_file):
        with open(vocab_file, 'r+') as f:
            word_index = json.load(f)
            source_vocab = word_index['source_data']
            target_vocab = word_index['target_data'] 

        print('Vocabulary loaded from',  vocab_file)
        return Vocabulary(source_vocab, target_vocab)

if __name__ == '__main__':
    source, target = Vocabulary.read_and_tokenize('github_train.txt', encoding_type='char')
    vocab = Vocabulary.build(source, target)
    vocab.save('vocab.json')
    vocab = Vocabulary.load('vocab.json')
