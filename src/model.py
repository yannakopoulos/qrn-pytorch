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
    r"""
    A position encoder that derives a single-vector representation for a
    sentence consisting of a list of embedding vectors x, taking into account
    each word's position in the sentence. Calculates output representation m as
    follows:

        m = \sum_j l_j \circ x_j

    where

        l is a column vector where l_{kj} = (1 - j/J) - (k/d) (1 - 2j/J)
        x is a list of J word embeddings
        d is the dimension of the embedding space
        \circ is element-wise multiplication

    For more information, see
        Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus.
        End-To-End Memory Networks. In NIPS, 2015.
    """
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
    r"""
    A single QRN unit. Accepts input in the form of a statement x_t, a
    question q_t, and the previous hidden layer h_{t-1}. Calculates output
    hidden layer h_t as follows:

        z_t = \alpha(x_t, q_t) = \sigma(W^{(z)}(x_t \circ q_t) + b^{(z)})

        \tilde{h}_t = \rho(x_t, q_t) = \tanh(W^{(h)}[x_t; q_t] + b^{(h)})

        h_t = z_t \tilde{h}_t + (1 - z_t) h_{t-1}

    where

        \alpha is an update gate
        \rho is a reduce function
        \sigma is the sigmoid activation
        \tanh is the hyperbolic tangent activation
        W^{(z)} and W^{(h)} are weight matrices
        b^{(z)} and b^{(h)} are bias vectors
        \circ is element-wise multiplication
        [;] is vector concatenation along the row

    For more information, see
        Minjoon Seo, Sewon Min, Ali Farhadi, and Hannaneh Hajishirzi.
        Query-Reduction Networks for Question Answering. In ICLR, 2017.
    """
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
    r"""
    A QRN model consisting of multiple QRN units chained together in layers,
    optionally bidirectional. Accepts as input a story (list of tensors
    representing statements) and a question (single tensor).

    Layers are stacked by passing the outputs of the previous layer as inputs
    to the current layer. Only the question is passed through the QRN; the
    story remains unchanged. If the ORN is not bidirectional, the question
    input to the next layer is given by

        q_t^{k+1} = h_t^{k}

    If the QRN is bidirectional, the question input to the next layer is
    instead given by

        q_t^{k+1} = \overrightarrow{h}_t^k + \overleftarrow{h}_t^k

    where

        \overrightarrow{h} represents the hidden layer in the forward direction
        \overleftarrow{h} represents the hidden layer in the backward direction

    For more information, see
        Minjoon Seo, Sewon Min, Ali Farhadi, and Hannaneh Hajishirzi.
        Query-Reduction Networks for Question Answering. In ICLR, 2017.
    """
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
            if self.bidirectional:
                h_states_backward.append(
                    [FloatTensor(self.hidden_size).fill_(0)])

        for i in range(len(story)):
            q_states[0].append(question)

        for layer in range(self.n_layers):
            # forward pass
            for i, statement in enumerate(story):
                h_forward = self.forward_layers[layer](
                    statement, q_states[layer][i],
                    h_states_forward[layer][i])
                h_states_forward[layer].append(h_forward)

            # backward pass
            if self.bidirectional:
                for i, statement in enumerate(reversed(story)):
                    h_backward = self.backward_layers[layer](
                        statement, q_states[layer][-(i + 1)],
                        h_states_backward[layer][i])
                    h_states_backward[layer].append(h_backward)

            # calculate question for next layer if necessary
            if layer < self.n_layers - 1:
                if self.bidirectional:
                    for i in range(len(story)):
                        q_states[layer + 1].append(
                            h_states_forward[layer][i + 1] +
                            h_states_backward[layer][-(i + 1)])
                else:
                    for i in range(len(story)):
                        q_states[layer + 1].append(
                            h_states_forward[layer][i + 1])

        return h_states_forward[-1][-1]


class Model(nn.Module):
    r"""
    A QRN model similar to the one discussed in
        Minjoon Seo, Sewon Min, Ali Farhadi, and Hannaneh Hajishirzi.
        Query-Reduction Networks for Question Answering. In ICLR, 2017.

    Accepts as input a story (list of tensors representing statements) and a
    question (single tensor). Consists of
        * an embedding layer with dimension equal to the hidden layer size that
            encodes each word as a vector
        * a positional encoder that encodes each statement as a single vector
            by combining its word vectors
        * a QRN that outputs a hidden layer representing the network's
            predicted answer to the question
        * a final linear layer that decodes this prediction back into a
            single word
    """
    # TODO: test/make work on batch data
    def __init__(self, n_words, hidden_size, n_layers, bidirectional=False):
        super(Model, self).__init__()
        self.embed = nn.Embedding(n_words + 1, hidden_size, padding_idx=0)
        self.encoder = PositionalEncoder(hidden_size)
        self.QRN = QRN(hidden_size, hidden_size, n_layers, bidirectional)
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
