#!/usr/bin/env python3
import os
import re
from model import LongTensor


def tokenize(line):
    r"""
    Tokenizes a line of text into words and punctuation.

    Parameters
    ----------
    line : str
        Line of text to tokenize.

    Returns
    -------
    tokens : list(str)
        List of tokens in line.
    """
    tokens = [
        x.strip().lower() for x in re.split(r"(\W+)+", line) if x.strip()]
    return tokens


def load_data(path, tasks=None):
    r"""
    Loads training/validation data from the bAbI dataset.

    Parameters
    ----------
    path : str
        Path to folder containing bAbI corpus files of the form
        `qa{task}_train.txt`
    tasks : list(int) or "all"
        Task(s) to load training data from. If task(s) not given, loads all
        available tasks.

    Returns
    -------
    stories : list(list(list(str)))
        A list of stories. Each story is a list of statements and each
        statement is a list of tokens.
    questions : list(list(str))
        A list of questions corresponding to each story. Each question is a
        list of tokens.
    answers : list(str)
        A list of answers corresponding to each story and question. Each answer
        is a single word.
    """
    if tasks is None:
        files = [os.path.join(path, file) for file in os.listdir(path)]
    else:
        files = [
            os.path.join(path, file) for file in os.listdir(path)
            if int(re.search(r"qa(\d+)", file).group(1)) in tasks]

    stories = []
    questions = []
    answers = []

    for file in files:
        with open(file, "r") as f:
            for line in f:
                ID, line = line.split(" ", 1)
                ID = int(ID)
                if ID == 1:
                    # start of a new story
                    story = [tokenize(line)]
                elif "\t" in line:
                    # question and answer line
                    question, answer, _ = line.split("\t")
                    question = tokenize(question)
                    substory = [
                        ["{}:".format(i + 1)] + line
                        for i, line in enumerate(story)]
                    stories.append(substory)
                    questions.append(["{}:".format(len(story) + 1)] + question)
                    answers.append(answer)
                else:
                    # regular line
                    story.append(tokenize(line))

    return stories, questions, answers


def data_to_index(data, dictionary):
    r"""
    Converts all tokens in data to embedding indices according to a dictionary.

    Parameters
    ----------
    data : tuple(list(list(list(int))), list(list(int)), list(int))
        Data tuple of stories, questions, and answers. Each story is a
        list of statements and each statement is a list of tokens. Similarly,
        each question is a list of tokens. Each answer is a single token.
    dictionary : dict(str: int)
        Dictionary to convert tokens to indices.

    Returns
    -------
    stories : list(list(tensor(int)))
        A list of stories. Each story is a list of statements and each
        statement is a tensor of embedding indices representing tokens.
    questions : list(tensor(int))
        A list of questions corresponding to each story. Each question is a
        tensor of embedding indices representing tokens.
    answers : tensor(int)
        A list of answers corresponding to each story and question. Each answer
        is a tensor with a single embedding indiex representing a token.
    """
    stories, questions, answers = data
    stories = [[
        LongTensor([dictionary.get(word, 0) for word in statement])
        for statement in story]
        for story in stories]
    questions = [
        LongTensor([dictionary.get(word, 0) for word in question])
        for question in questions]
    answers = [LongTensor([dictionary.get(answer, 0)]) for answer in answers]
    return stories, questions, answers


def load_corpus(path, tasks=None):
    r"""
    Loads bAbI corpus.

    Parameters
    ----------
    path : str
        Path to folder containing bAbI corpus files of the form
        `qa{task}_train.txt`
    tasks : list(int) or "all"
        Task(s) to load training data from. If task(s) not given, loads all
        available tasks.

    Returns
    -------
    train_data : tuple(list(list(tensor(int))), list(tensor(int)), tensor(int))
        Training data tuple of stories, questions, and answers. Each story is a
        list of statements and each statement is a tensor of ints representing
        the embedding index of a token. Similarly, each question is a tensor of
        ints representing the embedding index of a token. Each answer is a
        single int representing the embedding index of a one-word answer. The 0
        index is reserved for a padding embedding.
    test_data : tuple(list(list(tensor(int))), list(tensor(int)), tensor(int))
        Same as above, but for testing data.
    n_words : int
        Number of unique tokens in the training corpus.
    """
    train_data = load_data(os.path.join(path, "train"), tasks=tasks)
    test_data = load_data(os.path.join(path, "test"), tasks=tasks)

    vocab = set([
        word for story in train_data[0]
        for statement in story
        for word in statement])
    vocab.update([word for question in train_data[1] for word in question])
    vocab.update(train_data[2])
    dictionary = {word: i + 1 for i, word in enumerate(vocab)}
    n_words = len(vocab)

    train_data = data_to_index(train_data, dictionary)
    test_data = data_to_index(test_data, dictionary)

    return train_data, test_data, n_words
