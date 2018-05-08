from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors

from subjectivity.model import SubjectivityPredictor
from subjectivity.utils import is_subjective, convert_text_into_vector_sequence


class SubjectivityClassifier(object):
    def __init__(self, model_filename, word_filename):
        self._word_model = KeyedVectors.load_word2vec_format(word_filename)
        self._subj_model = SubjectivityPredictor.load(model_filename)

    def classify_sentences_in_text(self, text):
        '''

        :param text: The document that needs to be classified
        :return: A dict with two keys:
                 'subjective': the list of subjective sentences in the text
                 'objective': the list of objective sentences in the text
        '''
        sentences_list = sent_tokenize(self.__sanitize_text(text))
        subjective_sentences = []
        objective_sentences = []
        for sentence in sentences_list:
            sentence = self.__clean_sentence(sentence)
            if not sentence:
                continue
            prediction = self._subj_model.predict(convert_text_into_vector_sequence(self._word_model, sentence))
            if prediction == is_subjective:
                subjective_sentences.append(sentence)
            else:
                objective_sentences.append(sentence)
        return {'subjective': subjective_sentences, 'objective': objective_sentences}

    def __sanitize_text(self, text):
        text = text.replace('\n', '.\n')
        text = text.replace('.[', '. [')
        text = text.replace('.[', '. [')
        text = text.replace('...', '.')
        text = text.replace('..', '.')
        text = text.replace('\n.', '\n')
        return text

    def __clean_sentence(self, text):
        text = text.replace('\n', '')
        if text == '.': return ''
        return text
