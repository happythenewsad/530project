import torch
import torch.nn as nn
from torch.autograd import Variable


# don't touch this file... no reason to.

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

    def predict(self, vectors):
        print('\n> %s' % input_line)
        predictions = []
        for v in vectors:
            output = evaluate(Variable(lineToTensor(input_line)))
            # Get top N categories
            topv, topi = output.data.topk(1, 1, True)
            value = topv[0][1]
            category_index = topi[0][1]
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
        return predictions
