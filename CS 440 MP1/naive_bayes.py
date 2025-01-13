# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
import numpy as np
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels

"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=1.4, pos_prior=0.8, silently=False):
    print_values(laplace, pos_prior)

    yhats = []

    positiveProbability = Counter()
    negativeProbability = Counter()

    for i, single in zip(train_set, train_labels):
        if single == 1:
            positiveProbability.update(i)
        else:
            negativeProbability.update(i)

    total_positive = sum(positiveProbability.values())
    total_negative = sum(negativeProbability.values())

    vocabulary_size = len(set(positiveProbability) | set(negativeProbability))

    for single in dev_set:
        logPositive = math.log(pos_prior)
        logNegative = math.log(1 - pos_prior)

        for singleWord in single:
            logPositive += math.log(
                (positiveProbability[singleWord] + laplace) / (total_positive + vocabulary_size * laplace))
            logNegative += math.log(
                (negativeProbability[singleWord] + laplace) / (total_negative + vocabulary_size * laplace))

        yhats.append(int(logPositive > logNegative))

    return yhats