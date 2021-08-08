import numpy as np
import pandas as pd
from keras import backend as K
from keras import layers, models


def load_data(data_dir):
    total = pd.read_csv(data_dir)

    question = total["Q"].to_list()
    answer = total["A"].to_list()

    return question, answer


def data_preprocessing(question, answer):
    # 문장 벡터화
    input_texts = []
    target_texts = []

    input_characters = set()
    target_characters = set()

    for i in range(0, len(question)):
        input_text, target_text = question[i], answer[i]

        target_text = '\t' + target_text + '\n'

        input_texts.append(input_text)
        target_texts.append(target_text)

        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)

        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))

    data_preprocessing_dataset = {"input_texts": input_texts, "target_texts": target_texts,
                                  "input_characters": input_characters, "target_characters": target_characters}

    return data_preprocessing_dataset


def text_to_vector(data_preprocessing_dataset):
    input_characters = data_preprocessing_dataset["input_characters"]
    target_characters = data_preprocessing_dataset["target_characters"]

    # 문자 -> 숫자 변환용 사전
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    # 숫자 -> 문자 변환용 사전
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    text_to_vector_dataset = {"input_token_index": input_token_index, "target_token_index": target_token_index,
                              "reverse_input_char_index": reverse_input_char_index, "reverse_target_char_index": reverse_target_char_index}

    return text_to_vector_dataset


def RepeatVectorLayer(rep, axis):
    return layers.Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis), rep, axis),
                         lambda x: tuple((x[0],) + x[1:axis] + (rep,) + x[axis:]))


def main():
    latent_dim = 256

    base_dir = "../.."

    model_dir = "./attention_seq2seq.h5"
    data_dir = base_dir + "/Data/chat_data.csv"

    question, answer = load_data(data_dir)
    data_preprocessing_dataset = data_preprocessing(question, answer)
    text_to_vector_dataset = text_to_vector(data_preprocessing_dataset)

    num_encoder_tokens = len(data_preprocessing_dataset["input_characters"])
    num_decoder_tokens = len(data_preprocessing_dataset["target_characters"])

    max_encoder_seq_length = max([len(txt) for txt in data_preprocessing_dataset["input_texts"]])
    max_decoder_seq_length = max([len(txt) for txt in data_preprocessing_dataset["target_texts"]])

    input_token_index = text_to_vector_dataset["input_token_index"]
    target_token_index = text_to_vector_dataset["target_token_index"]
    reverse_target_char_index = text_to_vector_dataset["reverse_target_char_index"]


    # 인코더 생성
    encoder_inputs = layers.Input(shape=(max_encoder_seq_length, num_encoder_tokens))
    encoder = layers.GRU(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)

    # 디코더 생성.
    decoder_inputs = layers.Input(shape=(max_decoder_seq_length, num_decoder_tokens))
    decoder = layers.GRU(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _ = decoder(decoder_inputs, initial_state=state_h)

    # 어텐션 매커니즘.
    repeat_d_layer = RepeatVectorLayer(max_encoder_seq_length, 2)
    repeat_d = repeat_d_layer(decoder_outputs)

    repeat_e_layer = RepeatVectorLayer(max_decoder_seq_length, 1)
    repeat_e = repeat_e_layer(encoder_outputs)

    concat_for_score_layer = layers.Concatenate(axis=-1)
    concat_for_score = concat_for_score_layer([repeat_d, repeat_e])

    dense1_t_score_layer = layers.Dense(latent_dim // 2, activation='tanh')
    dense1_score_layer = layers.TimeDistributed(dense1_t_score_layer)
    dense1_score = dense1_score_layer(concat_for_score)

    dense2_t_score_layer = layers.Dense(1)
    dense2_score_layer = layers.TimeDistributed(dense2_t_score_layer)
    dense2_score = dense2_score_layer(dense1_score)
    dense2_score = layers.Reshape((max_decoder_seq_length, max_encoder_seq_length))(dense2_score)

    softmax_score_layer = layers.Softmax(axis=-1)
    softmax_score = softmax_score_layer(dense2_score)

    repeat_score_layer = RepeatVectorLayer(latent_dim, 2)
    repeat_score = repeat_score_layer(softmax_score)

    permute_e = layers.Permute((2, 1))(encoder_outputs)
    repeat_e_layer = RepeatVectorLayer(max_decoder_seq_length, 1)
    repeat_e = repeat_e_layer(permute_e)

    attended_mat_layer = layers.Multiply()
    attended_mat = attended_mat_layer([repeat_score, repeat_e])

    context_layer = layers.Lambda(lambda x: K.sum(x, axis=-1), lambda x: tuple(x[:-1]))
    context = context_layer(attended_mat)
    concat_context_layer = layers.Concatenate(axis=-1)
    concat_context = concat_context_layer([context, decoder_outputs])
    attention_dense_output_layer = layers.Dense(latent_dim, activation='tanh')
    attention_output_layer = layers.TimeDistributed(attention_dense_output_layer)
    attention_output = attention_output_layer(concat_context)
    decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(attention_output)


    # 모델 생성
    model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.load_weights(model_dir)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()


    # 추론(테스트)
    encoder_model = models.Model(encoder_inputs, [encoder_outputs, state_h])
    encoder_outputs_input = layers.Input(shape=(max_encoder_seq_length, latent_dim))
    decoder_inputs = layers.Input(shape=(1, num_decoder_tokens))
    decoder_state_input_h = layers.Input(shape=(latent_dim,))
    decoder_outputs, decoder_h = decoder(decoder_inputs, initial_state=decoder_state_input_h)
    repeat_d_layer = RepeatVectorLayer(max_encoder_seq_length, 2)
    repeat_d = repeat_d_layer(decoder_outputs)
    repeat_e_layer = RepeatVectorLayer(1, axis=1)
    repeat_e = repeat_e_layer(encoder_outputs_input)
    concat_for_score_layer = layers.Concatenate(axis=-1)
    concat_for_score = concat_for_score_layer([repeat_d, repeat_e])

    dense1_score_layer = layers.TimeDistributed(dense1_t_score_layer)
    dense1_score = dense1_score_layer(concat_for_score)

    dense2_score_layer = layers.TimeDistributed(dense2_t_score_layer)
    dense2_score = dense2_score_layer(dense1_score)
    dense2_score = layers.Reshape((1, max_encoder_seq_length))(dense2_score)

    softmax_score_layer = layers.Softmax(axis=-1)
    softmax_score = softmax_score_layer(dense2_score)

    repeat_score_layer = RepeatVectorLayer(latent_dim, 2)
    repeat_score = repeat_score_layer(softmax_score)
    permute_e = layers.Permute((2, 1))(encoder_outputs_input)
    repeat_e_layer = RepeatVectorLayer(1, axis=1)
    repeat_e = repeat_e_layer(permute_e)
    attended_mat_layer = layers.Multiply()
    attended_mat = attended_mat_layer([repeat_score, repeat_e])
    context_layer = layers.Lambda(lambda x: K.sum(x, axis=-1),
                                  lambda x: tuple(x[:-1]))

    context = context_layer(attended_mat)
    concat_context_layer = layers.Concatenate(axis=-1)
    concat_context = concat_context_layer([context, decoder_outputs])
    attention_output_layer = layers.TimeDistributed(attention_dense_output_layer)
    attention_output = attention_output_layer(concat_context)
    decoder_att_outputs = decoder_dense(attention_output)
    decoder_model = models.Model([decoder_inputs, decoder_state_input_h, encoder_outputs_input],
                                 [decoder_outputs, decoder_h, decoder_att_outputs])
    decoder_model.summary()


    def decode_sequence(input_seq):
        # 입력 문장을 인코딩
        enc_outputs, states_value = encoder_model.predict(input_seq)

        # 디코더의 입력으로 쓸 단일 문자
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # 첫 입력은 시작 문자인 '\t'로 설정
        target_seq[0, 0, target_token_index['\t']] = 1.

        # 문장 생성
        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            # 이전의 출력, 상태를 디코더에 넣어서 새로운 출력, 상태를 얻음
            # 이전 문자와 상태로 다음 문자와 상태를 얻는다고 보면 됨.
            dec_outputs, h, output_tokens = decoder_model.predict([target_seq, states_value, enc_outputs])

            # 사전을 사용해서 원 핫 인코딩 출력을 실제 문자로 변환
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # 종료 문자가 나왔거나 문장 길이가 한계를 넘으면 종료
            if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # 디코더의 다음 입력으로 쓸 데이터 갱신
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = h

        return decoded_sentence


    test_q, test_a = load_data(base_dir + "/Data/chat_test.csv")
    data_preprocessing_dataset = data_preprocessing(test_q, test_a)

    test_input_texts = data_preprocessing_dataset["input_texts"]
    test_target_texts = data_preprocessing_dataset["target_texts"]

    encoder_input_data = np.zeros((len(test_input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(test_input_texts, test_target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.


    for seq_index in range(200):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('"{}" -> "{}"'.format(test_input_texts[seq_index], decoded_sentence.strip()))


if __name__ == '__main__':
    main()