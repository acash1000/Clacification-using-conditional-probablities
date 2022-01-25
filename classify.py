import os
import math


# These first two functions require os operations and so are completed for you
# Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d + "/"
        files = os.listdir(directory + subdir)
        for f in files:
            bow = create_bow(vocab, directory + subdir + f)
            dataset.append({'label': label, 'bow': bow})
    return dataset


# Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d + '/'
        files = os.listdir(directory + subdir)
        for f in files:
            with open(directory + subdir + f, 'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])


# The rest of the functions need modifications ------------------------------
# Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # loops through the file
    with open(filepath, "r") as a_file:
        for line in a_file:
            stripped_line = line.strip()
            # if it is in vocab add it else assign to none
            if (stripped_line in vocab):
                if (stripped_line in bow):
                    bow[stripped_line] = bow[stripped_line] + 1
                else:
                    bow[stripped_line] = 1
            else:
                if (None in bow):
                    bow[None] = bow[None] + 1
                else:
                    bow[None] = 1

    return bow


# Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob = {}
    labelFrequ = {}
    sum = 0
    # TODO: add your code here
    # goes through training data list
    for i in label_list:
        counter = 0
        for j in training_data:
            # if lable is a match
            if (j['label'] == i):
                # add to the counter
                counter += 1
        labelFrequ[i] = counter + smooth
        sum += (counter + smooth)
    # the probability
    for i in label_list:
        logprob[i] = math.log(labelFrequ[i] / sum)
    return logprob


# Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1  # smoothing factor
    word_prob = {}
    # TODO: add your code here
    counter = 0
    dictionary = {}
    wordcount = 0
    #  iterates through the list training data
    for j in training_data:
        #   if it this dictionarys label is equal to the queried one
        if (j['label'] == label):
            # itterates through the bag of words
            for word in j['bow']:
                # adds it to the dictionary
                if (word in dictionary):
                    dictionary[word] = dictionary[word] + j['bow'].get(word)
                    wordcount += j['bow'].get(word)
                else:
                    dictionary[word] = j['bow'].get(word)
                    wordcount += j['bow'].get(word)
    for word in vocab:
        # to add words that didnt apear in the document but still in vocab
        if (word not in dictionary):
            dictionary[word] = 0
    #   the math for every one
    for i in dictionary:
        top = ((dictionary[i] + smooth * 1))
        bottom = wordcount + smooth * (len(vocab) + 1)
        word_prob[i] = math.log(top / bottom)

    return word_prob


##################################################################################
# Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)  # gets the label list
    vocab = create_vocabulary(training_directory, cutoff) # creates vocab
    training_data = load_training_data(vocab, training_directory) # instantiates training data
    prior_prob = prior(training_data, label_list) # gets prior prob
    prob_of_word_2016 = p_word_given_label(vocab, training_data, '2016') # gets conditional prob
    prob_of_word_2020 = p_word_given_label(vocab, training_data, '2020') # gets conditional prob
    retval['vocabulary'] = vocab
    retval['log prior'] = prior_prob
    retval['log p(w|y=2016)'] = prob_of_word_2016
    retval['log p(w|y=2020)'] = prob_of_word_2020
    return retval


# Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    # assigning all of the varibles from the model and the container varibles for the summation
    prior_prob = model['log prior']
    prob_of_word_2016 = model['log p(w|y=2016)']
    prob_of_word_2020 = model['log p(w|y=2020)']
    mult_2020 = 0
    mult_2016 = 0
    vocab = model['vocabulary']
    with open(filepath, "r") as a_file:  # opens the file
        # iterates through the file
        for word in a_file:
            word = word.strip()
            #   if the word is in the vocab
            if (word in vocab):
                mult_2020 += prob_of_word_2020.get(word)
                mult_2016 += prob_of_word_2016.get(word)
            #   if it is not in the vocab
            else:
                mult_2020 += prob_of_word_2020.get(None)
                mult_2016 += prob_of_word_2016.get(None)
    # adding the prior prob after the summation
    mult_2016 = mult_2016 + prior_prob.get('2016')
    mult_2020 = mult_2020 + prior_prob.get('2020')
    ## finding the max using a helper function
    ans = max(mult_2020, mult_2016)
    if (ans == mult_2020):
        foo = 2020
    else:
        foo = 2016
    retval['predicted y'] = foo
    retval['log p(y=2016|x)'] = mult_2016
    retval['log p(y=2020|x)'] = mult_2020
    return retval


def max(a, b):
    if a > b:
        return a
    else:
        return b


model = train('./corpus/training/', 2)
print(classify(model, './corpus/test/2016/0.txt'))
