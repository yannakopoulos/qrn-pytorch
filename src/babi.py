#!/usr/bin/env python3
import os
import re


def tokenize(line):
    """
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
    tokens = [x.strip() for x in re.split("(\W+)+", line) if x.strip()]
    return tokens


def load_data(path, tasks=None):
    """
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
    data : list(tuple(list(list(str)), list(str), str))
        A list of story, question, answer tuples. The story is a list of
        statements and each statement is itself a list of tokens. The question
        is similarly a list of tokens. The answer is a single token.
    """

    if tasks is None:
        files = [os.path.join(path, file) for file in os.listdir(path)]
    else:
        files = [
            os.path.join(path, file) for file in os.listdir(path)
            if int(re.search(r"qa(\d+)", file).group(1)) in tasks]

    data = []

    for file in files:
        with open(file, "r") as f:
            for line in f:
                ID, line = line.split(" ", 1)
                ID = int(ID)
                if ID == 1:
                    # start of a new story
                    story = []
                elif "\t" in line:
                    # question and answer line
                    question, answer, _ = line.split("\t")
                    question = tokenize(question)
                    substory = [
                        ["{}:".format(i + 1)] + line
                        for i, line in enumerate(story)]
                    data.append((substory, question, answer))
                else:
                    # regular line
                    story.append(tokenize(line))

    return data
