import tensorflow as tf
import tensorflow_hub as hub

# def build_model(max_len=512):
#     """build the model"""
#     #setting up the input layer
#     input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
#     input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
#     segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

#     #fetching the beat model
#     bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2")

#     #
#     pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
#     clf_output = sequence_output[:, 0, :]
#     net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
#     net = tf.keras.layers.Dropout(0.2)(net)
#     net = tf.keras.layers.Dense(32, activation='relu')(net)
#     net = tf.keras.layers.Dropout(0.2)(net)
#     out = tf.keras.layers.Dense(5, activation='softmax')(net)
    
#     model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
#     model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
#     return model
    
# text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
# preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
# encoder_inputs = preprocessor(text_input)

class Model:
    preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")

    def __init__(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        encoder_inputs = self.preprocessor(text_input)
        encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2", trainable=True)
        outputs = encoder(encoder_inputs)
        net = tf.keras.layers.Dense(outputs['pooled_output'].shape[1], activation='relu', dtype=tf.float32)(outputs['pooled_output'])
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, dtype=tf.float32)(net)
        self.model = tf.keras.Model(text_input, net)

    def train(self, train_data, test_labels, init_lr=0.001, epochs=5):
        pass

    def evaluate(self, test_data, test_labels):
        pass
        


if __name__ == '__main__':
    model = Model()
    tf.keras.utils.plot_model(model)