# https://github.com/spro/practical-pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import random
from helpers import *
from model import *
# from generate import *

all_categories = [0, 1, 2]                        


def random_training_pair():
    category = random.choice(all_categories)
    line = random.choice(category_lines_train[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor, decoder):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(line_tensor.size()[0]):
        print("{}\n{}\n{}".format(c, category_tensor[c], hidden))
        output, hidden = decoder(line_tensor[c], hidden)
    
    loss += criterion(output, line_tensor[c])

    loss.backward()
    decoder_optimizer.step()

    return output, loss.data[0]

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
