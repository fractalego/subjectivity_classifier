import os
import sys
from gensim.models import KeyedVectors

from subjectivity.model import SubjectivityPredictor
from subjectivity.train import train
from subjectivity.test import test
from subjectivity.utils import is_subjective, is_objective, get_data_from_list

_bucket_size = 10
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save')
_subjective_filename = os.path.join(_path, '../data/subj_dataset/subjective_cv_train.txt')
_objective_filename = os.path.join(_path, '../data/subj_dataset/objective_cv_train.txt')
_word_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/word_embeddings/glove.6B.50d.txt'))
_num_folds = 10
_max_epoch = 40


def _flatten(lst):
    return [item for sublist in lst for item in sublist]


def get_cross_validation_folds(subj_datalist, obj_datalist, num_folds=10):
    chunk_size = int(len(subj_datalist) / num_folds)
    subj_chunks = [subj_datalist[i:i + chunk_size] for i in range(0, len(subj_datalist), chunk_size)]
    obj_chunks = [obj_datalist[i:i + chunk_size] for i in range(0, len(obj_datalist), chunk_size)]
    folds = []
    for i in range(len(subj_chunks)):
        fold = {}
        fold['train'] = {'subjective': _flatten(subj_chunks[0:i]) + _flatten(subj_chunks[i + 1:]),
                         'objective': _flatten(obj_chunks[0:i]) + _flatten(obj_chunks[i + 1:])}
        fold['test'] = {'subjective': subj_chunks[i],
                        'objective': obj_chunks[i]}
        folds.append(fold)
    return folds


def get_average_precision_recall_f1_per_epoch(results, num_folds):
    total_recall = {}
    total_precision = {}
    total_f1 = {}
    for result in results:
        for epoch in result.keys():
            try:
                total_recall[epoch] += result[epoch]['recall']
                total_precision[epoch] += result[epoch]['precision']
                total_f1[epoch] += result[epoch]['f1']
            except:
                total_recall[epoch] = result[epoch]['recall']
                total_precision[epoch] = result[epoch]['precision']
                total_f1[epoch] = result[epoch]['f1']
    recall = {}
    precision = {}
    f1 = {}
    keys = list(total_recall.keys())
    for epoch in keys:
        recall[epoch] = total_recall[epoch] / num_folds
        precision[epoch] = total_precision[epoch] / num_folds
        f1[epoch] = total_f1[epoch] / num_folds
    return precision, recall, f1


def train_all_folds(folds, max_epoch):
    results = []
    for fold_index, fold in enumerate(folds):
        sys.stderr.write('Using Fold ' + str(fold_index + 1) + ' of ' + str(len(folds)) + '\n')
        model = SubjectivityPredictor(dropout=0.7)
        result = {}
        for epoch_num in range(max_epoch):
            sys.stderr.write('Training Epoch ' + str(epoch_num + 1) + ' of ' + str(max_epoch) + '\n')

            datalist = get_data_from_list(fold['train']['subjective'], is_subjective, _word_model)
            datalist += get_data_from_list(fold['train']['objective'], is_objective, _word_model)
            train(datalist, model, _saving_dir, epochs=1, bucket_size=10)

            datalist = get_data_from_list(fold['test']['subjective'], is_subjective, _word_model)
            datalist += get_data_from_list(fold['test']['objective'], is_objective, _word_model)
            result[epoch_num] = test(datalist, model)
        results.append(result)
    return results


if __name__ == '__main__':
    subj_datalist = open(_subjective_filename, encoding="ISO-8859-1").readlines()
    obj_datalist = open(_objective_filename, encoding="ISO-8859-1").readlines()
    folds = get_cross_validation_folds(subj_datalist, obj_datalist, num_folds=_num_folds)
    results = train_all_folds(folds, max_epoch=_max_epoch)
    precision, recall, f1 = get_average_precision_recall_f1_per_epoch(results, num_folds=_num_folds)
    for epoch in precision.keys():
        print('Epoch:', epoch)
        print('precision', precision[epoch])
        print('recall', recall[epoch])
        print('f1', f1[epoch])
