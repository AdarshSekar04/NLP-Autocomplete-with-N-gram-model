#This program is trained on a twitter dataset, for autocomplete.
#We use the n-gram model to predict the next token of a tweet, given n-1 characters.




'''
First, get all imports required. 
We use nltk to get our corpora for the model.
math for any mathematical computations.
random so we can intialize random numbers.
numpy for vectors and matrices.
pandas for reading in data.
'''

import math
import random
import numpy as np
import pandas as pd
import nltk
nltk.data.path.append('.')


'''
Here, we open our twitter data, and check the data we read in.
'''
with open("en_US.twitter.txt", "r") as f:
    data = f.read()
# print("Data type:", type(data))
# print("Number of letters:", len(data))
# print("First 300 letters of the data")
# print("-------")
# print(data[0:300])
# print("-------")

# print("Last 300 letters of the data")
# print("-------")
# print(data[-300:])
# print("-------")


'''
Now, we need to preprocess the data. For that, we have some helper functions.
1. First, we split the sentences by \n characters, and remove starting and trailing spaces.
2. Second, we need to tokenize our sentences. Must convert all words to lower case, and then tokenize them
'''


'''
This function splits our data into a list of sentences.
'''
def split_to_sentences(data):
    sentences = data.split("\n")
    sentences = [s.strip() for s in sentences] #Remove starting and trailing spaces, if any
    sentences = [s for s in sentences if len(s) > 0] #Removes empty strings
    return sentences


#Simple test on our preprocessed data.
# x = "I have a pen.\nI have an apple. \nAh\nApple pen.\n"
# sentences = split_to_sentences(x)
# print(sentences)


'''
This function tokenizes our sentences, and returns a list of lists of words.
Each sublist of the list returned is a sentence.
'''

def tokenize_sentences(sentences):
    #Initialize an empty array. We will add tokenized sentences to this array, and return it.
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)
    return tokenized_sentences

#Test on small example
# tokenized_sentences = tokenize_sentences(sentences)
# print(tokenized_sentences)

#This function simply calls the functions split_to_sentences, and tokenize sentences, and returns a list of lists of tokens, where each sublist is a sentence.

def get_tokenized_data(data):
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences

def get_tokenized_data2():
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences
#Small test 
# x = "I have a pen.\nI have an apple. \nAh\nApple pen.\n"
# tokenized = get_tokenized_data(x)
# print(tokenized)


#Now, we're going to call this on our actual data and split the data
tokenized_data = get_tokenized_data(data)
#This is used to mix up our data, and then split it into training and testing data
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]

#Print out splits
# print("{} data are split into {} train and {} test set".format(
#     len(tokenized_data), len(train_data), len(test_data)))

# print("First training sample:")
# print(train_data[0])
      
# print("First test sample")
# print(test_data[0])


'''
Now, we focus on getting the count of our words. First, we iterate through our training data, and get the count of all our words.
Then, we remove our Out-Of-Vocabulary words with <UNK>.
'''


'''
The below function takes in tokenized sentences, and returns a dictionary of token to count mapping.
'''
def count_words(tokenized_sentences):
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            # If the token is not in the dictionary yet, set the count to 1
            if token not in word_counts: 
                word_counts[token] = 1
            
            # If the token is already in the dictionary, increment the count by 1
            else:
                word_counts[token] += 1


    return word_counts

#Small test
# tokenized_sentences = [['sky', 'is', 'blue', '.'],
#                        ['leaves', 'are', 'green', '.'],
#                        ['roses', 'are', 'red', '.']]
# counts = count_words(tokenized_sentences)
# print(counts)

'''
Now, we'll get rid of any OOV's. OOV's are words the occur fewer than N times.
'''

def get_words_with_n_plus_frequency(tokenized_sentences, n_frequency):
    closed_vocab = []
    word_counts = count_words(tokenized_sentences)

    for word, count in word_counts.items():
        if count >= n_frequency:
            closed_vocab.append(word)
    return closed_vocab

#Small test
# tokenized_sentences = [['sky', 'is', 'blue', '.'],
#                        ['leaves', 'are', 'green', '.'],
#                        ['roses', 'are', 'red', '.']]
# tmp_closed_vocab = get_words_with_n_plus_frequency(tokenized_sentences, 2)
# print(f"Closed vocabulary:")
# print(tmp_closed_vocab)


'''
Now, we will replace our OOV words in our corpora with <unk>
'''
def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    #Convert our vocabulary to a set (for faster access of words)
    vocabulary = set(vocabulary)

    #This list will be the same tokenizes sentences, with OOV's replaced by <unk>
    #This is what we'll return
    replaced_tokenized_sentences = []

    for sentence in tokenized_sentences:
        replaced_sentence = []
        for token in sentence:
            if token not in vocabulary:
                replaced_sentence.append(unknown_token)
            else:
                replaced_sentence.append(token)
        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences

#Small test
# tokenized_sentences = [["dogs", "run"], ["cats", "sleep"]]
# vocabulary = ["dogs", "sleep"]
# tmp_replaced_tokenized_sentences = replace_oov_words_by_unk(tokenized_sentences, vocabulary)
# print(f"Original sentence:")
# print(tokenized_sentences)
# print(f"tokenized_sentences with less frequent words converted to '<unk>':")
# print(tmp_replaced_tokenized_sentences)

'''
Now, we simply call our get_words_with_n_plus_frequency function and replace_oov_words_by_unk function in a parent function.
Note: We use the same vocabulary for both train and test. THIS IS VERY IMPORTANT!!
'''

def preprocess_data(train_data, test_data, count_threshold):
    vocabulary = get_words_with_n_plus_frequency(train_data, count_threshold)
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary)
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary)
    return train_data_replaced, test_data_replaced, vocabulary
    

def preprocess_sentence(sentence, vocabulary):
    replaced_sentence = replace_oov_words_by_unk(sentence, vocabulary)
    return replaced_sentence

#Small test
# tmp_train = [['sky', 'is', 'blue', '.'],
#      ['leaves', 'are', 'green']]
# tmp_test = [['roses', 'are', 'red', '.']]

# tmp_train_repl, tmp_test_repl, tmp_vocab = preprocess_data(tmp_train, 
#                                                            tmp_test, 
#                                                            count_threshold = 1)

# print("tmp_train_repl")
# print(tmp_train_repl)
# print()
# print("tmp_test_repl")
# print(tmp_test_repl)
# print()
# print("tmp_vocab")
# print(tmp_vocab)

#Preprocess our train and test data
minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, test_data, 2)

# print("First preprocessed training sample:")
# print(train_data_processed[0])
# print()
# print("First preprocessed test sample:")
# print(test_data_processed[0])
# print()
# print("First 10 vocabulary:")
# print(vocabulary[0:10])
# print()
# print("Size of vocabulary:", len(vocabulary))

'''
Now, we can develop our n-gram model.
'''

def count_n_grams(data, n, start_token = '<s>', end_token = '<e>'):
    #Initialize a dictionary, which will store an n_gram and it's counts
    n_grams = {}

    for sentence in data:
        #Prepend the sentence with start_tokens, and append a single end_token at the end
        sentence = [start_token for i in range(0, n-1)] + sentence + [end_token]
        
        #Convert the sentence to a tuple, so we can use it as a key in our n_grams dict
        sentence = tuple(sentence)

        for i in range(0, len(sentence)-n + 1):
            n_gram = sentence[i:i+n]
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    
    return n_grams

#Small test
# sentences = [['i', 'like', 'a', 'cat'],
#              ['this', 'dog', 'is', 'like', 'a', 'cat']]
# print("Uni-gram:")
# print(count_n_grams(sentences, 1))
# print("Bi-gram:")
# print(count_n_grams(sentences, 2))


'''
Now, we estimate the probability of a word given the prior n-words.
This function takes in the word we want to guess, 
the previous_n_gram (n-1 gram), 
the n_gram_counts(n-1 gram),
the n_plus_1_gram_counts(n gram we are trying to predict),
the vocabulary_size,
and a constant k, for K-smoothing (prevents 0 probability)
'''
def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus_1_gram_counts, vocabulary_size, k = 1):
    #First, we will convert the previous_n_gram to a tuple, so we can get it's counts
    previous_n_gram = tuple(previous_n_gram)

    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)

    denominator = previous_n_gram_count + (k*vocabulary_size)

    n_plus_1_gram = previous_n_gram + (word,)
    n_plus_1_gram_count = n_plus_1_gram_counts.get(n_plus_1_gram, 0)

    numerator = n_plus_1_gram_count + k

    probability = numerator / denominator
    return probability

#Small test
# test your code
# sentences = [['i', 'like', 'a', 'cat'],
#              ['this', 'dog', 'is', 'like', 'a', 'cat']]
# unique_words = list(set(sentences[0] + sentences[1]))

# unigram_counts = count_n_grams(sentences, 1)
# bigram_counts = count_n_grams(sentences, 2)
# tmp_prob = estimate_probability("cat", "a", unigram_counts, bigram_counts, len(unique_words), k=1)

# print(f"The estimated probability of word 'cat' given the previous n-gram 'a' is: {tmp_prob:.4f}")
#Estimated should be 0.3333


'''
Now we create a function that calculates the probability of all possible words, given n previous words.
We then autocomplete by suggesting the most probable of all the possible words.
Function takes in:
the previous_n_gram (a list),
n_gram_counts (a dictionary),
n_plus_1_gram_counts (a dictionary),
vocabulary (a list of our vocab),
the smoothing constant k (an integer)
'''

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus_1_gram_counts, vocabulary, k = 1):
    #Convert the previous_n_gram to a tuple

    previous_n_gram = tuple(previous_n_gram)

    #Add <e> and <unk> to the vocabulary, as they could be the next word.
    #We don't add <s>, as it can't ever be the next word

    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)

    #Create a dictionary that will store the probability of every word being the next one.
    #We will be returning this
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, n_gram_counts, n_plus_1_gram_counts, vocabulary_size, k)
        probabilities[word] = probability
    return probabilities

#Small test
# sentences = [['i', 'like', 'a', 'cat'],
#              ['this', 'dog', 'is', 'like', 'a', 'cat']]
# unique_words = list(set(sentences[0] + sentences[1]))
# unigram_counts = count_n_grams(sentences, 1)
# bigram_counts = count_n_grams(sentences, 2)
# print(estimate_probabilities("a", unigram_counts, bigram_counts, unique_words, k=1))

# trigram_counts = count_n_grams(sentences, 3)
# print(estimate_probabilities(["<s>", "<s>"], bigram_counts, trigram_counts, unique_words, k=1))



'''
Now, to help visualize, lets create a count matrix.
This function will take in the n_plus_1_gram_counts(a dictionary), and the vocabulary(a list).
Returns a pandas DataFrame
'''

def make_count_matrix(n_plus_1_gram_counts, vocabulary):
    #Add the <e> token and <unk> token to our vocabulary
    vocabulary  = vocabulary + ["<e>", "<unk>"]

    #obtain unique n-grams
    n_grams = []
    for n_plus_1_gram in n_plus_1_gram_counts.keys():
        n_gram = n_plus_1_gram[:-1]
        n_grams.append(n_gram)
    
    #Remove duplicates
    n_grams = list(set(n_grams))

    #Map from n_gram to row
    row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}

    #Map next word to column
    col_index = {word: j for j, word in enumerate(vocabulary)}

    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol)) 
    for n_plus1_gram, count in n_plus_1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix

#Small test
# sentences = [['i', 'like', 'a', 'cat'],
#                  ['this', 'dog', 'is', 'like', 'a', 'cat']]
# unique_words = list(set(sentences[0] + sentences[1]))
# bigram_counts = count_n_grams(sentences, 2)

# print('bigram counts')
# print(make_count_matrix(bigram_counts, unique_words))


'''
Create a helper function that simply calls our make probability matrix function.
'''
def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix

#Small test
# sentences = [['i', 'like', 'a', 'cat'],
#                  ['this', 'dog', 'is', 'like', 'a', 'cat']]
# unique_words = list(set(sentences[0] + sentences[1]))
# print("trigram probabilities")
# trigram_counts = count_n_grams(sentences, 3)
# print(make_probability_matrix(trigram_counts, unique_words, k=1))


'''
Now, we calculate the perplexity, to see how accurate our model is.

This functions takes as arguments:
sentence - a list of strings,
n_gram_counts - a dictionary of n_gram counts,
n_plus_1_gram_counts - a dictionary of n+1-gram counts,
vocab_size - an integer,
k - smoothing constant
'''


def calculate_perplexity(sentence, n_gram_counts, n_plus_1_gram_counts, vocab_size, k = 1.0):
    #Get the length of the previous words
    n = len(list(n_gram_counts.keys())[0])

    #Prepend the sentence with <s>, and append a single <e>
    sentence = ["<s>"] * (n-1) + sentence + ["<e>"]

    N = len(sentence)

    #initialize the product_pi. (Using log-probability)
    log_sum = 0

    for i in range(n, N):

        n_gram = sentence[i-n:i]
        word = sentence[i]

        #Estimate the probability of the word
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus_1_gram_counts, vocab_size, k)

        log_sum += math.log(probability)

    perplexity = -1/N * log_sum
    return perplexity

#Small test
# sentences = [['i', 'like', 'a', 'cat'],
#                  ['this', 'dog', 'is', 'like', 'a', 'cat']]
# unique_words = list(set(sentences[0] + sentences[1]))

# unigram_counts = count_n_grams(sentences, 1)
# bigram_counts = count_n_grams(sentences, 2)


# perplexity_train1 = calculate_perplexity(sentences[0],
#                                          unigram_counts, bigram_counts,
#                                          len(unique_words), k=1.0)
# print(f"Perplexity for first train sample: {perplexity_train1:.4f}")

# test_sentence = ['i', 'like', 'a', 'dog']
# perplexity_test = calculate_perplexity(test_sentence,
#                                        unigram_counts, bigram_counts,
#                                        len(unique_words), k=1.0)
# print(f"Perplexity for test sample: {perplexity_test:.4f}")



'''
Now, we can build our autocomplete system. 
'''

'''
First, we will create a function that will return the most probable word and it's probability given:
the previous_tokens (a list of tokens),
n_gram_counts (a dictionary of n-grams to counts),
n_plus_1_gram_counts (a dictionary of n+1-grams to counts),
vocabulary (a list of words),
k, our smoothing factor,
start_with, the prefix of the string.
'''

def suggest_a_word(previous_tokens, n_gram_counts, n_plus_1_gram_counts, vocabulary, k=1.0, start_with=None):
    n = len(list(n_gram_counts.keys())[0])
    #Get the previous n tokens
    previous_n_gram = previous_tokens[-n:]

    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus_1_gram_counts, vocabulary, k)

    #initialize our suggestion to None, and our max_prob to 0
    suggestion = None
    max_prob = 0
    for word, prob in probabilities.items():
        if start_with:
            first_chars = len(start_with)
            if len(word) < first_chars or word[:first_chars] != start_with:
                continue
        if prob > max_prob:
            suggestion = word
            max_prob = prob
    
    return suggestion, max_prob

#Small test
# sentences = [['i', 'like', 'a', 'cat'],
#              ['this', 'dog', 'is', 'like', 'a', 'cat']]
# unique_words = list(set(sentences[0] + sentences[1]))

# unigram_counts = count_n_grams(sentences, 1)
# bigram_counts = count_n_grams(sentences, 2)

# previous_tokens = ["i", "like"]
# tmp_suggest1 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0)
# print(f"The previous words are 'i like',\n\tand the suggested word is `{tmp_suggest1[0]}` with a probability of {tmp_suggest1[1]:.4f}")

# print()
# # test your code when setting the starts_with
# tmp_starts_with = 'c'
# tmp_suggest2 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0, start_with=tmp_starts_with)
# print(f"The previous words are 'i like', the suggestion must start with `{tmp_starts_with}`\n\tand the suggested word is `{tmp_suggest2[0]}` with a probability of {tmp_suggest2[1]:.4f}")


'''
Now, we can use interpolation to calculate the most probable word using weighted probabilities of multiple
n-gram models.
Eg: P(chocolate | John drinks) = 0.7 * trigram(chocolate | John drinks) + 0.2 bigram(chocolate | John) 
                                                                        + 0.1 unigram(chocolate)
'''

'''
This function returns the various suggestions n-gram probabilities for different n-gram models, given:
previous_tokens (a list of tokens),
n_gram_counts_list (a list of various ngrams),
vocabulary (a list of words),
k, our smoothing factor,
start_with, an optional prefix
'''

def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k = 1.0, start_with = None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(0, model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus_1_gram_counts = n_gram_counts_list[i+1]
        suggestion = suggest_a_word(previous_tokens, n_gram_counts, n_plus_1_gram_counts, vocabulary, k, start_with)
        suggestions.append(suggestion)
    return suggestions

#Small test
# sentences = [['i', 'like', 'a', 'cat'],
#              ['this', 'dog', 'is', 'like', 'a', 'cat']]
# unique_words = list(set(sentences[0] + sentences[1]))

# unigram_counts = count_n_grams(sentences, 1)
# bigram_counts = count_n_grams(sentences, 2)
# trigram_counts = count_n_grams(sentences, 3)
# quadgram_counts = count_n_grams(sentences, 4)
# qintgram_counts = count_n_grams(sentences, 5)

# n_gram_counts_list = [unigram_counts, bigram_counts, trigram_counts, quadgram_counts, qintgram_counts]
# previous_tokens = ["i", "like"]
# tmp_suggest3 = get_suggestions(previous_tokens, n_gram_counts_list, unique_words, k=1.0)

# print(f"The previous words are 'i like', the suggestions are:")
# print(tmp_suggest3)


'''
Full test: Can edit the code below to mess around
'''
#get ngrams of varying lengths
# n_gram_counts_list = []
# for n in range(1, 6):
#     print("Computing n-gram counts with n =", n, "...")
#     n_model_counts = count_n_grams(train_data_processed, n)
#     n_gram_counts_list.append(n_model_counts)

# #Test with some previous tokens
# previous_tokens = ["i", "am", "to"]
# tmp_suggest4 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

# print(f"The previous words are {previous_tokens}, the suggestions are:")
# print(tmp_suggest4)

# previous_tokens = ["i", "want", "to", "go"]
# tmp_suggest5 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

# print(f"The previous words are {previous_tokens}, the suggestions are:")
# print(tmp_suggest5)

# previous_tokens = ["hey", "how", "are"]
# tmp_suggest6 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

# print(f"The previous words are {previous_tokens}, the suggestions are:")
# print(tmp_suggest6)

# previous_tokens = ["hey", "how", "are", "you"]
# tmp_suggest7 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

# print(f"The previous words are {previous_tokens}, the suggestions are:")
# print(tmp_suggest7)

# previous_tokens = ["hey", "how", "are", "you"]
# tmp_suggest8 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with="d")

# print(f"The previous words are {previous_tokens}, the suggestions are:")
# print(tmp_suggest8)