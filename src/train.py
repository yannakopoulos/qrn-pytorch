#!/usr/bin/env python3
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils_data
from babi import load_corpus
from model import LongTensor, FloatTensor, Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", required=True, help="bAbI data directory")
    parser.add_argument(
        "-t", "--tasks", type=int, nargs="+", help="bAbI tasks to load")
    parser.add_argument(
        "-s", "--hidden_size", type=int, default=32,
        help="size of hidden layer")
    parser.add_argument(
        "-l", "--n_layers", type=int, default=1,
        help="number of QRN layers")
    parser.add_argument(
        "-o", "--optimizer", choices=("Adam", "SGD", "Adagrad"),
        default="Adam", help="optimizer to use for training")
    parser.add_argument(
        "-r", "--learning_rate", type=float, default=0.01,
        help="learning rate to use for training")
    parser.add_argument(
        "-e", "--n_epochs", type=int, default=100,
        help="number of epochs to use for training")
    # parser.add_argument(
    #     "-b", "--batch_size", type=int, default=32,
    #     help="batch size to use for training")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print status messages")
    args = parser.parse_args()

    # check argument sanity
    if not os.path.isdir(args.data):
        parser.error("{} is not a directory".format(args.data))

    # # determine if GPU is available
    if torch.cuda.is_available() and args.verbose:
        print("GPU found:", torch.cuda.get_device_name(0))

    if args.verbose:
        print("Reading bAbI corpus...")

    train_data, test_data, n_words = load_corpus(args.data, tasks=args.tasks)
    # train_stories, train_questions, train_answers = train_data
    # test_stories, test_questions, test_answers = train_data

    if args.verbose:
        print("Initializing model...")

    model = Model(n_words, args.hidden_size, args.n_layers)

    if torch.cuda.is_available():
        model.cuda()

    loss_function = nn.CrossEntropyLoss()

    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)

    if args.verbose:
        print("Training QRN...")

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(args.n_epochs):
        if args.verbose:
            print("Epoch [{}/{}]".format(epoch + 1, args.n_epochs))

        model.train()
        losses = []
        correct = 0
        total = 0
        start = time.time()

        for story, question, answer in zip(*train_data):
            output = model(story, question)

            loss = loss_function(output.unsqueeze(0), answer)
            losses.append(loss.item())

            _, prediction = output.max(0)

            total += 1
            if prediction == answer:
                correct += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - start

        train_loss.append(sum(losses) / len(losses))
        train_acc.append(correct / total)

        if args.verbose:
            print("{0:.4f} seconds elapsed".format(elapsed))
            print(
                "Training: Loss = {0:.4f}, Accuracy = {1:.4f}"
                .format(train_loss[-1], train_acc[-1]))

        # evaluate on testing data
        model.eval()
        losses = []
        correct = 0
        total = 0

        for story, question, answer in zip(*test_data):
            output = model(story, question)

            loss = loss_function(output.unsqueeze(0), answer)
            losses.append(loss.item())

            _, prediction = output.max(0)

            total += 1
            if prediction == answer:
                correct += 1

        test_loss.append(sum(losses) / len(losses))
        test_acc.append(correct / total)

        if args.verbose:
            print(
                "Testing: Loss = {0:.4f}, Accuracy = {1:.4f}"
                .format(test_loss[-1], test_acc[-1]))

        if test_acc[-1] == 1:
            break  # no need to continue as we're not testing on any more data

    if args.verbose:
        print("\nTraining finished!")

    filename = "QRN-" + "-".join([
        "{}={}".format(arg, getattr(args, arg)) for arg in vars(args)
        if arg != "verbose"])
    results_file = filename + ".tsv"
    model_file = filename + ".model"

    print("\nSaving results to {}...".format(results_file))
    with open(results_file, "w") as f:
        f.write("epoch\ttrain_loss\ttrain_acc\ttest_loss\ttest_acc\n")
        for epoch in range(args.n_epochs):
            f.write("{}\t{}\t{}\t{}\t{}\n".format(
                epoch, train_loss[epoch], train_acc[epoch], test_loss[epoch],
                test_acc[epoch]))

    print("Saving model to {}...\n".format(model_file))
    torch.save(model.state_dict(), model_file)
