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
    def __init__(self):
        super(PositionalEncoder, self).__init__()

    def forward(self, embeddings):
        encoding = torch.mean(embeddings, dim=0)
        return encoding


class QRNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(QRNCell, self).__init__()
        # TODO: implement bidirectionality

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
        q = q.clone()  # FIXME: this is only true for QRNs with 1 layer
        return q, h


class QRN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(QRN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.layers = []

        if n_layers > 1:
            raise ValueError("QRNs with <1 layers are not yet supported")

        for layer in range(n_layers):
            self.layers.append(QRNCell(input_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, story, question):
        q_states = []
        h_states = []

        for layer in range(self.n_layers):
            q_states.append([])
            # initialize h_0 to zero vector
            h_states.append([torch.zeros(self.hidden_size)])

        for statement in story:
            # first layer uses question directly
            q_states[0].append(question)
            for layer, module in enumerate(self.layers):
                q, h = module(
                    statement, q_states[layer][-1], h_states[layer][-1])
                if layer < self.n_layers - 1:
                    # next layer uses output question
                    # unless this layer is the last one
                    q_states[layer + 1].append(q)
                # next iteration of same layer uses output hidden layer
                h_states[layer].append(h)

        return h_states[-1][-1]


class Model(nn.Module):
    def __init__(self, n_words, n_dimensions, hidden_size, n_layers):
        super(Model, self).__init__()

        # add 1 extra embedding for padding
        self.embed = nn.Embedding(n_words + 1, n_dimensions)
        # TODO: do positional encoding instead of BOW embedding average
        self.encoder = PositionalEncoder()
        self.QRN = QRN(n_dimensions, hidden_size, n_layers)
        # convert final prediction back into embedding index
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
