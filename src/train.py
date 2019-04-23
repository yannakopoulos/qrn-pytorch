#!/usr/bin/env python3
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.utils.data as utils_data
from babi import load_corpus
from model import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", required=True, help="bAbI data directory")
    parser.add_argument(
        "-t", "--tasks", type=int, nargs="+",
        help="specific bAbI tasks to load")
    parser.add_argument(
        "-s", "--hidden_size", type=int, default=32,
        help="size of hidden layer")
    parser.add_argument(
        "-l", "--n_layers", type=int, default=2,
        help="number of QRN layers")
    parser.add_argument(
        "-b", "--bidirectional", action="store_true",
        help="make QRN bidirectional")
    parser.add_argument(
        "-o", "--optimizer", choices=("Adam", "SGD", "Adagrad"),
        default="Adam", help="optimizer to use for training")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.01,
        help="learning rate to use for training")
    parser.add_argument(
        "-e", "--n_epochs", type=int, default=100,
        help="number of epochs to use for training")
    parser.add_argument(
        "-r", "--results", help="directory to save results")
    # parser.add_argument(
    #     "-b", "--batch_size", type=int, default=32,
    #     help="batch size to use for training")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print status messages")
    args = parser.parse_args()

    # check argument sanity
    if not os.path.isdir(args.data):
        parser.error("{} is not a directory".format(args.data))

    if args.verbose:
        print("Arguments:")
        for arg in vars(args):
            print("{} = {}".format(arg, getattr(args, arg)))

    if torch.cuda.is_available() and args.verbose:
        print("GPU found:", torch.cuda.get_device_name(0))

    if args.verbose:
        print("Reading bAbI corpus...")

    train_data, test_data, n_words = load_corpus(args.data, tasks=args.tasks)

    if args.verbose:
        print("Initializing model...")

    model = Model(n_words, args.hidden_size, args.n_layers, args.bidirectional)

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
    completed = 0

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

        completed = epoch

        if test_acc[-1] == 1:
            break  # no need to continue as we're not testing on any more data

    if args.verbose:
        print("\nTraining finished!")

    filename = "QRN-" + "-".join([
        "{}={}".format(arg, getattr(args, arg)) for arg in vars(args)
        if arg != "data" and arg != "results" and arg != "verbose"])
    results_file = os.path.join(args.results, filename + ".tsv")
    model_file = os.path.join(args.results, filename + ".model")

    if not os.path.isdir(args.results):
        os.mkdir(args.results)

    print("\nSaving results to {}...".format(results_file))
    with open(results_file, "w") as f:
        epochs = completed if completed < args.n_epochs else args.n_epochs
        f.write("epoch\ttrain_loss\ttrain_acc\ttest_loss\ttest_acc\n")
        for epoch in range(epochs):
            f.write("{}\t{}\t{}\t{}\t{}\n".format(
                epoch, train_loss[epoch], train_acc[epoch], test_loss[epoch],
                test_acc[epoch]))

    print("Saving model to {}...\n".format(model_file))
    torch.save(model.state_dict(), model_file)
