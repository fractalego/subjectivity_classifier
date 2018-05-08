import nltk
import gensim.utils as utils

is_subjective = [1., 0.]
is_objective = [0., 1.]


def get_words(text):
    tokenizer = nltk.tokenize.TweetTokenizer()
    words = tokenizer.tokenize(utils.to_unicode(text))
    return words


def capitalize(word):
    return word[0].upper() + word[1:]


def low_case(word):
    return word[0].lower() + word[1:]


def infer_vector_from_word(model, word):
    vector = model['entity']
    try:
        vector = model[word]
    except:
        try:
            vector = model[capitalize(word)]
        except:
            try:
                vector = model[low_case(word)]
            except:
                pass
    return vector


def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def bin_data_into_buckets(data, batch_size):
    buckets = []
    size_to_data_dict = {}
    for item in data:
        sentence_length = len(item['sentence_vectors'])
        try:
            size_to_data_dict[sentence_length].append(item)
        except:
            size_to_data_dict[sentence_length] = [item]
    for key in size_to_data_dict.keys():
        data = size_to_data_dict[key]
        chunks = get_chunks(data, batch_size)
        for chunk in chunks:
            buckets.append(chunk)
    return buckets


def convert_text_into_vector_sequence(model, text):
    words = get_words(text)
    vectors = []
    for word in words:
        vectors.append(infer_vector_from_word(model, word))
    return vectors


def get_data_from_list(datalist, classification, model):
    data = []
    for item in datalist:
        try:
            single_data = {}
            single_data['sentence_vectors'] = convert_text_into_vector_sequence(model, item)
            single_data['classification'] = classification
            data.append(single_data)
        except Exception as e:
            print('Exception caught during getting the data: ' + str(e))
    return data
