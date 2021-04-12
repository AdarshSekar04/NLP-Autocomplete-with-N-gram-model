import autocomplete
import sys
import random

def main(argv):
    #print(argv)
    n_grams = int(argv[1])
    

    print("Parsing dataset...")
    #Now, get our training and test data
    tokenized_data = autocomplete.get_tokenized_data2()
    # random.seed(87)
    # random.shuffle(tokenized_data)

    # train_size = int(len(tokenized_data) * 0.8)
    # train_data = tokenized_data[0:train_size]
    # test_data = tokenized_data[train_size:]
    test_data = []
    train_data = tokenized_data
    # print("{} data are split into {} train and {} test set".format(
    # len(tokenized_data), len(train_data), len(test_data)))

    # print("First training sample:")
    # print(train_data[0])
        
    # print("First test sample")
    # print(test_data[0])
    print("Preprocessing the data...")
    minimum_freq = 2
    train_data_processed, test_data_processed, vocabulary = autocomplete.preprocess_data(train_data, test_data, minimum_freq)  
    print("Creating N-gram models...")
    n_gram_counts_list = []
    for n in range(1, n_grams+1):
        n_model_counts = autocomplete.count_n_grams(train_data_processed, n)
        n_gram_counts_list.append(n_model_counts)
    print("Models created!\n\n")
    while True:
        print("Please input an incomplete sentence of whole words only:")
        text = input("")
        #Now preprocess the text
        tokenized_text = autocomplete.get_tokenized_data(text)
        print(tokenized_text)
        preprocessed = autocomplete.preprocess_sentence(tokenized_text, vocabulary)
        #Now predict
        print(preprocessed)
        previous_tokens = preprocessed[0]
        suggestions = autocomplete.get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, 1)
        print("Suggestions are:")
        print(suggestions)
        print("\n")
if __name__ == "__main__":
    main(sys.argv)