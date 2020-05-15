import io
import pickle
import torch

from utils import Transfrom

from seq2seq import Encoder
from seq2seq import Decoder

class Text2song(object):
    def __init__(self):
        def Load_Vocab(file):
            with open(file, 'rb') as fd:
                _vocab = pickle.load(fd)
            return _vocab

        def Load_Parameters(file):
            with open(file, 'rb') as fd:
                parameters_dict = pickle.load(fd)
            return parameters_dict

        torch.manual_seed(1)
        torch.set_num_threads(4)
        en_vocab_dur_file = './en_vocab_dur.pkl'
        de_vocab_dur_file = './de_vocab_dur.pkl'

        encoder_dur_model_file = './encoder_dur.10.pt'
        decoder_dur_model_file = './decoder_dur.10.pt'

        en_vocab_key_file = './en_vocab.pkl'
        de_vocab_key_file = './de_vocab.pkl'

        encoder_key_model_file = './encoder.10.pt'
        decoder_key_model_file = './decoder.10.pt'
        hyper_parameters_file = './parameters_dict.pkl'
        self.en_vocab_key = Load_Vocab(en_vocab_key_file)
        self.de_vocab_key = Load_Vocab(de_vocab_key_file)

        self.en_vocab_dur = Load_Vocab(en_vocab_dur_file)
        self.de_vocab_dur = Load_Vocab(de_vocab_dur_file)

        self.trf_key = Transfrom(self.en_vocab_key)
        self.trf_dur = Transfrom(self.en_vocab_dur)

        self.parameters_dict = Load_Parameters(hyper_parameters_file)

        en_embedding_dim = self.parameters_dict['en_embedding_dim']
        de_embedding_dim = self.parameters_dict['de_embedding_dim']
        hidden_dim = self.parameters_dict['hidden_dim']
        num_layers = self.parameters_dict['num_layers']
        bidirectional = self.parameters_dict['bidirectional']
        use_lstm = self.parameters_dict['use_lstm']
        self.use_cuda_dur = self.use_cuda_key = False
        batch_size = 1
        dropout_p = 0.0

        self.encoder_key = Encoder(en_embedding_dim, hidden_dim, self.en_vocab_key.n_items, num_layers, dropout_p, bidirectional,
                          use_lstm,
                          self.use_cuda_key)
        self.decoder_key = Decoder(de_embedding_dim, hidden_dim, self.de_vocab_key.n_items, num_layers, dropout_p, bidirectional,
                          use_lstm,
                          self.use_cuda_key)
        self.encoder_dur = Encoder(en_embedding_dim, hidden_dim, self.en_vocab_dur.n_items, num_layers, dropout_p,
                              bidirectional,
                              use_lstm,
                              self.use_cuda_dur)
        self.decoder_dur = Decoder(de_embedding_dim, hidden_dim, self.de_vocab_dur.n_items, num_layers, dropout_p,
                              bidirectional,
                              use_lstm,
                              self.use_cuda_dur)

        self.encoder_key.load_state_dict(torch.load(encoder_key_model_file, map_location='cpu'))
        self.decoder_key.load_state_dict(torch.load(decoder_key_model_file, map_location='cpu'))
        self.encoder_dur.load_state_dict(torch.load(encoder_dur_model_file, map_location='cpu'))
        self.decoder_dur.load_state_dict(torch.load(decoder_dur_model_file, map_location='cpu'))

        self.encoder_key.eval()
        self.decoder_key.eval()
        self.encoder_dur.eval()
        self.decoder_dur.eval()

        """ __init__ return the parameters: {self.trf_dur,self.trf_key;
                                            self.encoder_dur,self.encoder_key;
                                            self.decoder_dur,self.decoder_key;
                                            self.en_vocab_dur,self.en_vocab_key;
                                            self.de_vocab_dur,self.de_vocab_key;
                                            self.use_cuda_dur,self,self.use_cuda_key.}"""

    def get_song(self, lyric):
        def stop_before_eos(li):
            if '_EOS_' in li:
                i = li.index('_EOS_')
                return li[:i]
            return li

        def important_function_in_while_loop(trf, sent, encoder, decoder, de_vocab, use_cuda, en_sent):
            en_seq, en_seq_len = trf.trans_input(sent)

            en_seq = torch.LongTensor(en_seq)
            encoder_input = en_seq
            encoder_output, encoder_state = encoder(encoder_input, en_seq_len)

            # initial decoder hidden
            decoder_state = decoder.init_state(encoder_state)

            # Start decoding
            decoder_inputs = torch.LongTensor([de_vocab.item2index['_START_']])

            pred_char = ''

            if use_cuda: decoder_inputs = decoder_inputs.cuda()
            decoder_outputs, decoder_state = decoder(decoder_inputs, encoder_output, decoder_state)

            max_len = len(en_sent.split())

            return (pred_char, encoder_output, decoder_outputs, decoder_state, max_len)

        f_en_test = io.StringIO(lyric)

        pred_list = []

        while True:
            en_sent = f_en_test.readline()

            if not en_sent: break

            sent = en_sent.strip()
            pred_sent_dur = []
            pred_sent_key = []
            pred_char_key, encoder_output_key, decoder_outputs_key, decoder_state_key, max_len_key = \
                important_function_in_while_loop(self.trf_key, sent, self.encoder_key, self.decoder_key, self.de_vocab_key, self.use_cuda_key,
                                                 en_sent)

            pred_char_dur, encoder_output_dur, decoder_outputs_dur, decoder_state_dur, max_len_dur = \
                important_function_in_while_loop(self.trf_dur, sent, self.encoder_dur, self.decoder_dur, self.de_vocab_dur, self.use_cuda_dur,
                                                 en_sent)

            # Greedy search
            while pred_char_key != '_EOS_' and pred_char_dur != '_EOS_':
                log_prob_key, v_idx_key = decoder_outputs_key.data.topk(1)
                pred_char_key = self.de_vocab_key.index2item[v_idx_key.item()]
                pred_sent_key.append(pred_char_key)

                log_prob_dur, v_idx_dur = decoder_outputs_dur.data.topk(1)
                pred_char_dur = self.de_vocab_dur.index2item[v_idx_dur.item()]
                pred_sent_dur.append(pred_char_dur)

                if (len(pred_sent_dur) > max_len_dur or len(pred_sent_dur) > max_len_key): break

                decoder_inputs_dur = torch.LongTensor([v_idx_dur.item()])
                if self.use_cuda_dur: decoder_inputs_dur = decoder_inputs_dur.cuda()
                decoder_outputs_dur, decoder_state_dur = self.decoder_dur(decoder_inputs_dur, encoder_output_dur,
                                                                     decoder_state_dur)

                decoder_inputs_key = torch.LongTensor([v_idx_key.item()])
                if self.use_cuda_key: decoder_inputs_key = decoder_inputs_key.cuda()
                decoder_outputs_key, decoder_state_key = self.decoder_key(decoder_inputs_key, encoder_output_key,
                                                                     decoder_state_key)

            pred_list.append(
                {'lyrics': sent, 'key': stop_before_eos(pred_sent_key), 'duration': stop_before_eos(pred_sent_dur)})
            # pred_list.append({'lyrics': sent, 'key': pred_sent_key, 'duration': pred_sent_dur})

        return pred_list

if __name__ == '__main__':
    lyric = open('./input.txt', 'r', encoding='utf-8').read()
    mySong = Text2song()
    pred_list = mySong.get_song(lyric)
    print(pred_list)

'''Text2Song:
  1. Model_file(seq2seq.py); Model_Parameters(*.pt;*.pkl) included;
  2. Text2Song.py: class.
          (i) __init__ :load the model and parameters;
          (ii) get_song(lyric):
          (iii) if __name__=="__main__": Test.'''