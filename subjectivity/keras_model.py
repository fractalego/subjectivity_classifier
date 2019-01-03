import tensorflow.keras as keras

_sentence_vocab_size = 50
_word_proj_size_for_rnn = 25
_hidden_dim = 200
_output_size = 2

_memory_dim = 100
_stack_dimension = 1


def get_keras_model(dropout):
    model = keras.models.Sequential([
        keras.layers.Dense(_word_proj_size_for_rnn, activation='relu', input_shape=(None, _sentence_vocab_size)),
        keras.layers.Bidirectional(keras.layers.GRU(_memory_dim)),
        keras.layers.Dense(_hidden_dim, activation='relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(_output_size, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    model = get_keras_model(dropout=0.7)
