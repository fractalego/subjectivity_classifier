import os
import random

from gensim.models import KeyedVectors

from subjectivity.model import SubjectivityPredictor
from subjectivity.utils import bin_data_into_buckets, get_data_from_list
from subjectivity.utils import is_objective, is_subjective

_bucket_size = 10
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save')
# _subjective_filename = os.path.join(_path, '../data/subj_dataset/subjective_train.txt')
# _objective_filename = os.path.join(_path, '../data/subj_dataset/objective_train.txt')
_subjective_filename = os.path.join(_path, '../data/subj_dataset/subjective_cv_train.txt')
_objective_filename = os.path.join(_path, '../data/subj_dataset/objective_cv_train.txt')
_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/word_embeddings/glove.6B.50d.txt'))


def train(data, model, saving_dir, epochs=20, bucket_size=10, trace_every=None):
    import sys

    buckets = bin_data_into_buckets(data, bucket_size)
    for i in range(epochs):
        random_buckets = sorted(buckets, key=lambda x: random.random())
        if trace_every:
            sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')
        for bucket in random_buckets:
            training_bucket = []
            for item in bucket:
                try:
                    sentence_vectors = item['sentence_vectors']
                    y = item['classification']
                    training_bucket.append((sentence_vectors, y))
                except Exception as e:
                    print('Exception caught during training: ' + str(e))
            if len(training_bucket) > 0:
                model.train(training_bucket, epochs=1)
        if trace_every and i % trace_every == 0:
            save_filename = saving_dir + '/subj-' + str(i) + '.tf'
            sys.stderr.write('Saving into ' + save_filename + '\n')
            model.save(save_filename)


if __name__ == '__main__':
    data = get_data_from_list(open(_objective_filename, encoding="ISO-8859-1").readlines(), is_objective, _model)
    data += get_data_from_list(open(_subjective_filename, encoding="ISO-8859-1").readlines(), is_subjective, _model)
    data = sorted(data, key=lambda x: random.random())

    nn_model = SubjectivityPredictor(dropout=0.7)
    train(data, nn_model, _saving_dir, epochs=39, bucket_size=10, trace_every=1)
