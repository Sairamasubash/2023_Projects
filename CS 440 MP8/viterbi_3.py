"""
Part 4: Here should be your best version of viterbi,
with enhancements such as dealing with suffixes/prefixes separately
"""

import math
from collections import defaultdict, Counter
from math import log
import sys
import numpy as np

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect

limitS = 75   # Here is the limiting of the suffixes.

newSmoothing = 1e-7   # Here is a new smoothing parameter.

# Here we are defining a function called training.
def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """

    dictionaryOne = {}   # Here we are creating an empty dictionary to store emission probabilities.

    dictionaryTwo = {}   # Here we are creating an empty dictionary to store transition probabilities.

    tList = []   # Here is a list to store unique tags encountered in the training data.


    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    # Here we are iterating through each sentence in the provided training data.
    for each in sentences:

        pTag = ''   # Here we are initializing previous tag to an empty string.

        # Here we are iterating through each word and its corresponding tag in a sentence.
        for idx, (W, T) in enumerate(each):

            # Here we are checking if the tag hasn't been added to the unique tag list.
            if T not in tList:

                tList.append(T)   # Here we are adding tag to unique tag list.

            # Here we are initializing or update word count for the given tag.
            dictionaryOne.setdefault(T, {}).setdefault(W, 0)

            # Here we are incrementing the count of the word for the given tag.
            dictionaryOne[T][W] += 1

            if idx > 0:   # Here we are checking if it's not the first word-tag pair in the sentence.

                # Here we are initializing or update transition count for the given tag sequence.
                dictionaryTwo.setdefault(pTag, {}).setdefault(T, 0)

                # Here we are incrementing the count of the current tag following the previous tag.
                dictionaryTwo[pTag][T] += 1

            pTag = T   # Here we are updating the previous tag to the current tag.

    # Here we are initializing three empty dictionaries.
    dictionaryFour, dictionaryFive, dictionarySix = {}, {}, {}

    # Here we are iterating over key-value pairs of dictionaryOne.
    for T, W2 in dictionaryOne.items():

        # Here we are creating a list of words from 'words' with count 1 that are not in dictionaryFour.
        listOne = [W for W, count in W2.items() if count == 1 and W not in dictionaryFour]

        # Here we are updating dictionaryFour with words from listOne and corresponding T values.
        dictionaryFour.update({W: T for W in listOne})

        # Here we are updating dictionaryFive with T as key and listOne as value.
        dictionaryFive[T] = listOne

    # Here we are creating a dictionary with T as key and length of words list as value.
    dictionarySix = {T: len(W3) for T, W3 in dictionaryFive.items()}

    # Here we are initializing a dictionary to store suffix values
    suffixValues = {

        # Here we are calculating the value of T using a for loop.
        T: [each for each, number in Counter([W[-2:] for W in W4]).items() if number >= limitS]

        # Here we are iterating over key-value pairs in dictionaryFive
        for T, W4 in dictionaryFive.items()

    }

    # Here we are defining a function called suffix_in_values.
    def suffix_in_values(W, value):

        # Here we are checking if any suffix of length 'i' from 'word' exists in 'values' and iterating through suffix lengths 2, 3, and 4.
        return any(W[-each:] in value for each in range(2, 5))

    # Here we are initializing a new dictionary called 'dictionaryEight'.
    dictionaryEight = {

        # Here we are calculating the value of T using a for loop.
        T: [W for W in W5 if not suffix_in_values(W, suffixValues[T])]

        # Here we are iterating over key-value pairs (T, words) in 'dictionaryFive'.
        for T, W5 in dictionaryFive.items()

    }

    # Here we are creating a dictionary with T as key and length of words list as value.
    dictionaryNine = {T: len(W6) for T, W6 in dictionaryEight.items()}

    # Here we are copying the contents of dictionaryNine to suffixProbabilityOne.
    suffixProbabilityOne = dictionaryNine.copy()

    # Here we are calculating the total of values in suffixProbabilityOne with a minor adjustment.
    totalSuffixes = sum(suffixProbabilityOne.values()) + newSmoothing * len(suffixProbabilityOne)

    # Here we are adjusting the values in suffixProbabilityOne by dividing each value with totalSuffixes.
    suffixProbabilityOne = {T: (value + newSmoothing) / totalSuffixes for T, value in suffixProbabilityOne.items()}

    # Here we are initializing dictionaryTen as a defaultdict of dicts.
    dictionaryTen = defaultdict(dict)

    # Here we are iterating over key-value pairs of dictionaryFive.
    for T, W7 in dictionaryFive.items():

        # Here we are iterating over words in words list.
        for W in W7:

            # Here we are iterating over integers from 2 to 4 inclusive.
            for each in range(2, 5):

                # Here we are checking if the suffix of the word is in suffixValues[T].
                if W[-each:] in suffixValues[T]:

                    # Here we are setting default value for key in dictionaryTen.
                    dictionaryTen[W[-each:]].setdefault(T, 0)

                    # Here we are incrementing the count of the suffix in dictionaryTen.
                    dictionaryTen[W[-each:]][T] += 1

    # Here we are initializing an empty dictionary to store probabilities for each suffix with a key of length two.
    suffixProbabilityTwo = {

        # Here we are calculating the probability of each 'T' by dividing its value by the total sum of all values in the inner dictionary for each key and its associated inner dictionary in 'dictionaryTen'.
        key: {T: value / total for T, value in inner.items()}

        # Here we are iterating over each key and its associated inner dictionary in 'dictionaryTen'.
        for key, inner in dictionaryTen.items()

        # Here we are calculating the total sum of all values in the inner dictionary and store it in a single-item list.
        for total in [sum(inner.values())]
    }

    # Here we are iterating through each tag in the emission dictionary.
    for T in dictionaryOne:

        # Here we are iterating through each word associated with the current tag.
        for W in dictionaryOne[T]:

            # Here we are applying smoothing to the word count for the current tag.
            dictionaryOne[T][W] += emit_epsilon

        # Here we are handling unknown words for the current tag.
        dictionaryOne[T]['UNKNOWN'] = emit_epsilon

    # Here we are adding an 'END' key to the transition dictionary to handle sentence terminations.
    dictionaryTwo['END'] = {}

    # Here we are iterating through each current tag in the transition dictionary.
    for CT in dictionaryTwo:

        for T in tList:   # Here we are iterating through each unique tag.

            # Here we are initializing or update transition count for the current tag to the next tag.
            dictionaryTwo[CT].setdefault(T, epsilon_for_pt)

            # Here we are applying smoothing to the transition count.
            dictionaryTwo[CT][T] += epsilon_for_pt

    # Here we are iterating through each current tag and its transitions in the dictionary.
    for CT, next_tags in dictionaryTwo.items():

        # Here we are computing total transitions from the current tag.
        total_transition = sum(next_tags.values())

        for NT in next_tags:   # Here we are iterating through each next tag.

            # Here we are converting transition count to log probability.
            dictionaryTwo[CT][NT] = np.log(dictionaryTwo[CT][NT] / total_transition)

    # Here we are iterating through each tag and its words in the emission dictionary.
    for T, words in dictionaryOne.items():

        # Here we are computing total emissions for the current tag.
        total_emission = sum(words.values())

        for W in words:   # Here we are iterating through each word.

            # Here we are converting word count to log probability.
            dictionaryOne[T][W] = np.log(dictionaryOne[T][W] / total_emission)

    # Here we are returning many values from the training data.
    return dictionaryOne, dictionaryTwo, tList, suffixProbabilityOne, suffixValues, suffixProbabilityTwo

# Here we are defining a function called viterbi_stepforward.
def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob, tList, suffixProbabilityOne, suffixValues, suffixProbabilityTwo):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """

    lProb = {}   # Here is dictionary to store the log probabilities for each tag at the current step.

    tagPrediction = {}   # Here is dictionary to store the predicted tag sequences leading up to the current step.

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    for T in tList:   # Here we are iterating over each possible tag in the tag list.

        # Here we are getting the emission probability of the observed word for the current tag T.
        if word not in emit_prob[T]:

            # Here is the default to the first suffix probability for tag T.
            newValue = suffixProbabilityOne[T]

            # Here we are checking if there are suffix values for tag T and the word's last two characters are in those suffix values.
            if suffixValues[T] and word[-2:] in suffixValues[T]:

                # Here we are updating the newValue to the maximum of the current value and the second suffix probability for the word's last two characters for tag T.
                newValue = max(newValue, suffixProbabilityTwo[word[-2:]][T])

            # Here we are calculating emission probability for unknown words by adding the log of the new value to the UNKNOWN emission probability for tag T.
            eProb = emit_prob[T]['UNKNOWN'] + np.log(newValue)

        # Here we are checking if the word is already in the emission probabilities for tag T.
        else:

            # Here we are assigning the emission probability of the word for tag T directly.
            eProb = emit_prob[T][word]

        # Here we are calculating the total probability for reaching the current tag T from each possible previous tag.
        probabilities = [

            # Here we are getting the probability of the previous tag.
            prev_prob[pTag] + eProb + trans_prob[pTag].get(T, math.log(epsilon_for_pt))

            for pTag in tList   # Here we are using a for loop.

        ]

        # Here we are getting the maximum probability from the list of calculated probabilities.
        max_prob = max(probabilities)

        # Here we are determining which previous tag resulted in the maximum probability for reaching the current tag T.
        best_previous_tag = tList[probabilities.index(max_prob)]

        lProb[T] = max_prob   # Here we are storing the maximum probability for reaching the current tag T.

        # Here we are storing the best predicted tag sequence leading up to the current tag T.
        tagPrediction[T] = prev_predict_tag_seq[best_previous_tag] + [T]

    # Here we are returning the calculated log probabilities and predicted tag sequences for the current step.
    return lProb, tagPrediction

# Here we are defining a function called viterbi_3.
def viterbi_3(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    # Here we are extracting many different values from the training data.
    eProb, tProb, tList, suffixProbabilityOne, suffixValues, suffixProbabilityTwo = get_probs(train)

    # Here we are defining a function called helperViterbi.
    def helperViterbi(sentence):

        # Here we are initializing log probabilities for tags (special handling for 'START' tag).
        lProb = {tag: (math.log(epsilon_for_pt) if tag != 'START' else 0) for tag in tList}

        # Here we are initializing tag predictions storage.
        tagPrediction = {tag: [] for tag in tList}

        # Here we are iterating through each word in the sentence.
        for idx, word in enumerate(sentence):

            # Here we are computing next step in the Viterbi algorithm.
            lProb, tagPrediction = viterbi_stepforward(idx, word, lProb, tagPrediction, eProb, tProb, tList, suffixProbabilityOne, suffixValues, suffixProbabilityTwo)

        return lProb, tagPrediction   # Here we are returning the computed log probabilities and tag predictions.

    # TODO:(III)
    # according to the storage of probabilities and sequences, get the final prediction.

    # Here we are generating final tag predictions for each sentence in the test data.
    allPredictions = [

        # Here we are pairing each word with its predicted tag, choosing the tag sequence with the highest log probability.
        list(zip(sentence, tagPrediction[max(lProb, key=lProb.get)]))

        for sentence in test   # Here we are iterating through each sentence in the test data.

        # Here we are computing log probabilities and tag predictions for the current sentence.
        for lProb, tagPrediction in [helperViterbi(sentence)]

    ]

    return allPredictions   # Here we are returning the final tag predictions for all sentences.