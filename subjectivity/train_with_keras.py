import os

from gensim.models import KeyedVectors

from subjectivity.data_generator import DataGenerator
from subjectivity.keras_model import get_keras_model
from subjectivity.utils import get_data_from_list
from subjectivity.utils import is_objective, is_subjective

_bucket_size = 10
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save')
# _subjective_filename = os.path.join(_path, '../data/subj_dataset/subjective_train.txt')
# _objective_filename = os.path.join(_path, '../data/subj_dataset/objective_train.txt')
_subjective_filename = os.path.join(_path, '../data/subj_dataset/subjective_cv_train.txt')
_objective_filename = os.path.join(_path, '../data/subj_dataset/objective_cv_train.txt')
_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/word_embeddings/glove.6B.50d.txt'))


def train(model, generator, saving_dir, epochs=20, trace_every=None):
    import sys

    for i in range(epochs):
        if trace_every:
            sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')

        model.fit_generator(generator=generator, epochs=1, verbose=1)
        if trace_every and i % trace_every == 0:
            save_filename = saving_dir + '/subj-keras-' + str(i) + '.tf'
            sys.stderr.write('Saving into ' + save_filename + '\n')
            model.save(save_filename)


if __name__ == '__main__':
    data = get_data_from_list(open(_objective_filename, encoding="ISO-8859-1").readlines(), is_objective, _model)
    data += get_data_from_list(open(_subjective_filename, encoding="ISO-8859-1").readlines(), is_subjective, _model)
    generator = DataGenerator(data, batch_size=20)
    nn_model = get_keras_model(dropout=0.7)

    train(nn_model, generator, _saving_dir, epochs=39, trace_every=1)
