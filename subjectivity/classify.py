import os

_path = os.path.dirname(__file__)

_error_message = '''
Please provide a text as an input.
You can either provide the text as an argument: python -m subjectivity.classify this is my opinion.
Or pipe the text from the command line: python -m subjectivity.classify < data/random_text.txt
'''


def _aggregate_sentence(args):
    return_str = ''
    for argument in args:
        return_str += argument + ' '
    return return_str


def _get_subj_or_obj_sentences_from_text(text):
    from subjectivity.subjectivity_classifier import SubjectivityClassifier
    classifier = SubjectivityClassifier(model_filename=os.path.join(_path, '../data/save/subj-29.tf'),
                                        word_filename=os.path.join(_path, '../data/word_embeddings/glove.6B.50d.txt'))
    return classifier.classify_sentences_in_text(text)


def print_sentences(sentences_dict):
    print('\nOBJECTIVE SENTENCES:')
    [print(item) for item in sentences_dict['objective']]
    print('\nSUBJECTIVE SENTENCES:')
    [print(item) for item in sentences_dict['subjective']]


if __name__ == '__main__':
    import os
    import sys

    if len(sys.argv) > 1:
        sentence = _aggregate_sentence(sys.argv[1:])
        print_sentences(_get_subj_or_obj_sentences_from_text(sentence))
    else:
        if os.isatty(0):
            print(_error_message)
            exit(0)
        sentence = sys.stdin.read().strip()
        if sentence:
            print_sentences(_get_subj_or_obj_sentences_from_text(sentence))
