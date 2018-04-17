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
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden


    def init_hidden(self):
        return Variable(torch.rand(1, self.hidden_size))

    def evaluate(self, line_tensor):
        hidden = self.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
        
        return output

    def predict(self, vectors):
        predictions = []
        for v in vectors:
            output = self.evaluate(Variable(torch.FloatTensor(v)))
            # Get top N categories
            topv, topi = output.data.topk(1, 1, True)
            value = topv[0][1]
            category_index = topi[0][1]
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
        return predictions
