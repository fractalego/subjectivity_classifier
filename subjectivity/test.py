import os
import random

from gensim.models import KeyedVectors
from subjectivity.model import SubjectivityPredictor
from subjectivity.utils import get_data_from_list
from subjectivity.utils import is_objective, is_subjective

_bucket_size = 10
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save')
_subjective_filename = os.path.join(_path, '../data/subj_dataset/subjective_test.txt')
_objective_filename = os.path.join(_path, '../data/subj_dataset/objective_test.txt')
_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/word_embeddings/glove.6B.50d.txt'))


def count_true_and_false_positives_and_negatives(prediction, expected):
    true_positives = sum(
        [prediction[i] == is_subjective and expected[i] == is_subjective for i in range(len(expected))])
    false_positives = sum(
        [prediction[i] == is_subjective and expected[i] != is_subjective for i in range(len(expected))])
    true_negatives = sum(
        [prediction[i] != is_subjective and expected[i] != is_subjective for i in range(len(expected))])
    false_negatives = sum(
        [prediction[i] != is_subjective and expected[i] == is_subjective for i in range(len(expected))])
    return true_positives, false_positives, true_negatives, false_negatives


def test(data, model):
    prediction = []
    expected = []
    for item in data:
        classification = item['classification']
        sentence_vectors = item['sentence_vectors']
        prediction.append(model.predict(sentence_vectors))
        expected.append(classification)
    true_positives, false_positives, true_negatives, false_negatives = \
        count_true_and_false_positives_and_negatives(prediction, expected)
    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * 1 / (1 / precision + 1 / recall)
        accuracy = (true_positives + true_negatives) \
                   / (true_positives + true_negatives + false_positives + false_negatives)
        return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
    except:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}


if __name__ == '__main__':
    data = get_data_from_list(open(_objective_filename, encoding="ISO-8859-1").readlines(), is_objective, _model)
    data += get_data_from_list(open(_subjective_filename, encoding="ISO-8859-1").readlines(), is_subjective, _model)
    data = sorted(data, key=lambda x: random.random())

    for i in range(0, 10):
        print('Epoch:', i)
        nn_model = SubjectivityPredictor.load(os.path.join(_saving_dir, 'subj-' + str(i) + '.tf'))
        result = test(data, nn_model)
        print('precision', result['precision'])
        print('recall', result['recall'])
        print('f1', result['f1'])
        print('accuracy', result['accuracy'])
