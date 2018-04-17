# https://github.com/spro/practical-pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from helpers import *
from model import *
# from generate import *



def random_training_set(chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    chunk = file[start_index:end_index]
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename("/rnn/pt_file"))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)




# Parse command line arguments
#argparser = argparse.ArgumentParser()
#argparser.add_argument('filename', type=str)
#argparser.add_argument('--n_epochs', type=int, default=2000)
#argparser.add_argument('--print_every', type=int, default=100)
#argparser.add_argument('--hidden_size', type=int, default=50)
#argparser.add_argument('--n_layers', type=int, default=2)
#argparser.add_argument('--learning_rate', type=float, default=0.01)
#argparser.add_argument('--chunk_len', type=int, default=200)
#args = argparser.parse_args()
#file, file_len = read_file(args.filename)
