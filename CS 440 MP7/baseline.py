"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

# Here we are defining a function to predict tags for test data based on word-tag frequency in the training data.
def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    wordFrequency = {}   # Here is a dictionary to store the frequency of each tag for a given word.

    tagFrequency = {}   # Here is a dictionary to store the frequency of each tag in the training data.

    # Here we are iterating over each sentence in the training data.
    for sentence in train:

        # Here we are iterating over each word-tag pair in the sentence.
        for W, T in sentence:

            # Here we are initializing the word-tag frequency if not present.
            wordFrequency.setdefault(W, {}).setdefault(T, 0)

            wordFrequency[W][T] += 1   # Here we are incrementing the tag frequency for the given word.

            tagFrequency[T] = tagFrequency.get(T, 0) + 1   # Here we are incrementing the overall tag frequency.

    # Here we are finding the most frequent tag in the training data.
    mostTag = max(tagFrequency, key=tagFrequency.get)

    # Here we are predicting tags for test data. If word is in wordFrequency, use the most frequent tag for that word. Otherwise, use the mostTag.
    result = [[(W, max(wordFrequency.get(W, {}), key=wordFrequency.get(W, {}).get, default=mostTag)) for W in sentence] for sentence in test]

    return result   # Here we are returning the predicted tags for the test data.