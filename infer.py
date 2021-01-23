import torch 
import torch.nn.functional as F 

class Inferencer:
    def __init__(self, model, vocab, device, max_steps=100):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_steps = max_steps

    def convert_to_ids(self, sentence, vocab_dict):
        token_ids = [vocab_dict.get(word, vocab_dict['<unk>']) for word in sentence]
        return torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)

    @staticmethod
    def reverse_dict(vocab_dict):
        ''' Constructs id -> word dictionary by reversing keys and values ''' 
        return {i : word for word, i in vocab_dict.items()}

    def infer_greedy(self, sentence):
        ''' Corrects a given sentence using a greedy decoding technique '''
        self.model.eval()

        input_tensor = self.convert_to_ids(sentence, self.vocab.source_vocab)
        ids_words_dict = self.reverse_dict(self.vocab.target_vocab)

        with torch.no_grad():
            encoded_input, source_mask = self.model.encode(input_tensor)

            greedy_hyp = ['<s>']

            for _ in range(self.max_steps):
                target = self.convert_to_ids(greedy_hyp, self.vocab.target_vocab)
                decoder_out = self.model.decode(target, encoded_input, source_mask)

                h_scores = F.log_softmax(decoder_out, dim=-1)
                max_id = h_scores.argmax(2)[:,-1].item()

                potential_token = ids_words_dict.get(max_id)
                if max_id != '<\s>':
                    greedy_hyp.append(potential_token)
                else:
                    break

        return ''.join(greedy_hyp[1:])