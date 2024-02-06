
# Attempt utilising LSTM Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class StockLSTM(nn.Module):
    def __init__(self):
        # , embedding_dim, hidden_dim, vocab_size, tagset_size
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 50, num_layers = 1, batch_first = True)
        self.linear = nn.Linear(50,1)

        #self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        #self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        '''
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        '''
        x,_ = self.lstm(x)
        x = self.linear(x)
        return x