# bigram_naive_bayes.py
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
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
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
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.007, bigram_laplace=0.007, bigram_lambda=0.3, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    positiveUnigramCounter = Counter()
    negativeUnigramCounter = Counter()
    positiveBigramCounter = Counter()
    negativeBigramCounter = Counter()

    total_positive = 0
    total_negative = 0

    for i, single in enumerate(train_set):
        for j in range(len(single)):
            singleWord = single[j]
            if train_labels[i] == 1:
                positiveUnigramCounter[singleWord] += 1
                total_positive += 1
            else:
                negativeUnigramCounter[singleWord] += 1
                total_negative += 1

            if j < len(single) - 1:
                bigramNaiveBayes = (single[j], single[j + 1])
                if train_labels[i] == 1:
                    positiveBigramCounter[bigramNaiveBayes] += 1
                else:
                    negativeBigramCounter[bigramNaiveBayes] += 1

    positiveProbability = sum(train_labels) / len(train_labels)
    negativeProbability = 1 - positiveProbability

    yhats = []

    for single in tqdm(dev_set, disable=silently):
        positiveLogPositive = math.log(positiveProbability)
        negativeLogPositive = math.log(negativeProbability)

        for j in range(len(single)):
            singleWord = single[j]

            positiveUnigramProbability = (positiveUnigramCounter[singleWord] + unigram_laplace) / (total_positive + unigram_laplace * len(positiveUnigramCounter))
            negativeUnigramProbability = (negativeUnigramCounter[singleWord] + unigram_laplace) / (total_negative + unigram_laplace * len(negativeUnigramCounter))

            positiveLogPositive += (1 - bigram_lambda) * math.log(positiveUnigramProbability)
            negativeLogPositive += (1 - bigram_lambda) * math.log(negativeUnigramProbability)

            if j < len(single) - 1:
                bigramNaiveBayes = (single[j], single[j + 1])
                positiveBigramProbability = (positiveBigramCounter[bigramNaiveBayes] + bigram_laplace) / (total_positive + bigram_laplace * len(positiveBigramCounter))
                negativeBigramProbability = (negativeBigramCounter[bigramNaiveBayes] + bigram_laplace) / (total_negative + bigram_laplace * len(negativeBigramCounter))

                positiveLogPositive += bigram_lambda * math.log(positiveBigramProbability)
                negativeLogPositive += bigram_lambda * math.log(negativeBigramProbability)

        if positiveLogPositive > negativeLogPositive:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats