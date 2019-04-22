#!/usr/bin/env python3
import math
import torch
import torch.nn as nn


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


class PositionalEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(PositionalEncoder, self).__init__()
        self.d = hidden_size

    def forward(self, embeddings):
        J = embeddings.shape[0]
        L = FloatTensor([[
            (1 - j/J) - (k/self.d) * (1 - 2*j/J) for k in range(1, self.d + 1)]
            for j in range(1, J + 1)])
        encoding = torch.sum(L * embeddings, dim=0)
        return encoding


class QRNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(QRNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.alpha = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Sigmoid())
        self.rho = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size), nn.Tanh())

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, q, h_prev):
        z = self.alpha(x * q)
        h_tilde = self.rho(torch.cat((x, q), 0))
        h = z * h_tilde + (1 - z) * h_prev
        return h


class QRN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, bidirectional=False):
        super(QRN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.forward_layers = []
        self.backward_layers = []

        for layer in range(n_layers):
            self.forward_layers.append(QRNCell(input_size, hidden_size))
            if torch.cuda.is_available():
                self.forward_layers[-1].cuda()
            if bidirectional:
                self.backward_layers.append(QRNCell(input_size, hidden_size))
                if torch.cuda.is_available():
                    self.backward_layers[-1].cuda()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, story, question):
        q_states = []
        h_states_forward = []
        h_states_backward = []

        for layer in range(self.n_layers):
            q_states.append([])
            # initialize h_0 to zero vector
            h_states_forward.append(
                [FloatTensor(self.hidden_size).fill_(0)])
            h_states_backward.insert(
                0, [FloatTensor(self.hidden_size).fill_(0)])

        # TODO: implement bidirectionality
        for statement in story:
            # first layer uses question directly
            q_states[0].append(question)
            for layer, module in enumerate(self.forward_layers):
                h_forward = module(
                    statement, q_states[layer][-1],
                    h_states_forward[layer][-1])
                if layer < self.n_layers - 1:
                    # same iteration of next layer uses q = h from this layer
                    # unless this layer is the last one
                    q_states[layer + 1].append(h_forward)
                # next iteration of same layer uses output hidden layer
                h_states_forward[layer].append(h_forward)

        return h_states_forward[-1][-1]


class Model(nn.Module):
    # TODO: test/make work on batch data
    def __init__(self, n_words, hidden_size, n_layers, bidirectional=False):
        super(Model, self).__init__()
        self.embed = nn.Embedding(n_words + 1, hidden_size, padding_idx=0)
        self.encoder = PositionalEncoder(hidden_size)
        self.QRN = QRN(hidden_size, hidden_size, n_layers)
        self.predict = nn.Linear(hidden_size, n_words + 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.embed.reset_parameters()
        self.QRN.reset_parameters()
        self.predict.reset_parameters()

    def forward(self, story, question):
        story_embed = []
        for statement in story:
            statement_embed = self.encoder(self.embed(statement))
            story_embed.append(statement_embed)
        question_embed = self.encoder(self.embed(question))
        output = self.QRN(story_embed, question_embed)
        output = self.predict(output)
        return output
