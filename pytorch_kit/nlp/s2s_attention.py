from __future__ import unicode_literals
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # input_size --> nums of words dict len (e.g. 4345)
        # hidden_size --> 256
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # [1, 1, 256]
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # output_size --> nums of words dict len (e.g. 2803)
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, enc_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), enc_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        
        return output, hidden, attn_weights
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.word2count = {}
        self.n_words = 2
    
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(lang1, lang2, reverse=False):
    lines = open('data/{}-{}.txt'.format(lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    pairs = filter_pairs(pairs)
    print(f'trimmed to {len(pairs)} sentence pairs')
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print(f'{input_lang.name}:', input_lang.n_words, 'words')
    print(f'{output_lang.name}:', output_lang.n_words, 'words')
    return input_lang, output_lang, pairs

def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, enc, dec,
          enc_optimizer, dec_optimizer,
          criterion, max_length=MAX_LENGTH, teacher_forcing_ratio=0.5):
    
    # encoder
    enc_hidden = enc.init_hidden()
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    enc_outputs = torch.zeros(max_length, enc.hidden_size, device=device)
    loss = 0

    for enc_i in range(input_length):
        enc_output, enc_hidden = enc(input_tensor[enc_i], enc_hidden)
        enc_outputs[enc_i] = enc_output[0, 0]

    # decoder
    dec_input = torch.tensor([[SOS_token]], device=device)
    dec_hidden = enc_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if True:
        for dec_i in range(target_length):
            dec_output, dec_hidden, dec_attention = dec(dec_input, dec_hidden, enc_outputs)
            loss += criterion(dec_output, target_tensor[dec_i])
            dec_input = target_tensor[dec_i]
    else:
        for dec_i in range(target_length):
            dec_output, dec_hidden, dec_attention = dec(dec_input, dec_hidden, enc_outputs)
            topv, topi = dec_output.topk(1)
            dec_input = topi.squeeze().detach()

            loss += criterion(dec_output, target_tensor[dec_i])
            if dec_input.item() == EOS_token:
                break

    loss.backward()
    enc_optimizer.step()
    dec_optimizer.step()
    return loss.item() / target_length

def evaluate(enc, dec, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        enc_hidden = enc.init_hidden()
        enc_outputs = torch.zeros(max_length, enc.hidden_size, device=device)

        for enc_i in range(input_length):
            enc_output, enc_hidden = enc(input_tensor[enc_i], enc_hidden)
            enc_outputs[enc_i] += enc_output[0,0]
        
        dec_input = torch.tensor([[SOS_token]], device=device)
        dec_hidden = enc_hidden
        
        dec_words = []
        dec_attentions = torch.zeros(max_length, max_length)

        for dec_i in range(max_length):
            dec_output, dec_hidden, dec_attention = dec(dec_input, dec_hidden, enc_outputs)
            dec_attentions[dec_i] = dec_attention.data
            topv, topi = dec_output.data.topk(1)
            if topi.item() == EOS_token:
                dec_words.append('<EOS>')
                break
            else:
                dec_words.append(output_lang.index2word[topi.item()])
            dec_input = topi.squeeze().detach()
        
        return dec_words, dec_attentions[:dec_i+1]

def train_iters(input_lang, output_lang, pairs, enc, dec, n_iters, print_every=1000, learning_rate=.01):
    print_loss_total = 0

    enc_optimizer = optim.SGD(enc.parameters(), lr=learning_rate)
    dec_optimizer = optim.SGD(dec.parameters(), lr=learning_rate)
    train_pairs = [tensors_from_pair(input_lang, output_lang, random.choice(pairs)) for _ in range(n_iters)]

    criterion = nn.NLLLoss()

    for it in range(1, n_iters + 1):
        training_pair = train_pairs[it-1]
        input_tensor = training_pair[0]    # e.g. [7, 1]
        target_tensor = training_pair[1]   # e.g. [5, 1]

        loss = train(input_tensor, target_tensor, enc, dec, enc_optimizer, dec_optimizer, criterion)

        break

        print_loss_total += loss

        if it % print_every == 0:
            print_loss_total /= print_every
            print('iters: {:>5d}/{}, loss: {:.6f}'.format(it, print_loss_total))
            print_loss_total = 0


def main():
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
    # print(random.choice(pairs))

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    train_iters(input_lang, output_lang, pairs, encoder1, attn_decoder1, 75000, print_every=5000)

if __name__ == '__main__':
    main()