## Requirements
```bash
pip install -r requirements.txt
```

## Train
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove*.zip
python binary_pre_training.py
```

## Flow
- load_data : 학습 및 테스트 데이터 불러오기 (CSV 파일을 기준으로 코드 작성).

- pre_processing
  - 문장 전처리하는 함수.  
  - tokenizer를 사용해 문장을 단어로 분리한 후, 단어를 dictionary의 index 값으로 바꿔준다. ( "1: 'i', 2: 'a', 3: 'o', 4: 'he' ... )
  - train_x, test_x, val_x 데이터 중 가장 길이가 긴 문장의 token 개수를 구한다.
  - pad_sequence를 사용해 가장 긴 문장의 token 개수만큼 0 값으로 채운다. (train_x, text_x, val_x 의 모든 문장의 shape 형태가 같아지게 된다.)
  - example) "Why do I hate life and people while other people don't do that nearly as frequently?" -> [  13   12    1   17    9  133    2    8   35  197    3   42   35   12   12   17  953   44 1558   71    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    ... ...     0    0    0    0    0    0   ]

- text_to_vector
  - dictionary에 등록되어 있는 index들을 pre-trained에 등록되어 있는 값으로 mapping 시켜주는 함수.
  - 4: 'he' 라는 단어로 등록되어 있다. glove에 등록되어 있는 he라는 단어 값을 가져와 embedding 값으로 mapping한다.
  - 4: [-1.45669997e-01 -6.80280030e-01 -4.74730015e-01 -1.32550001e-01 -6.98290020e-02 -1.02609999e-01  4.03360009e-01 ... ...   -1.26709998e-01 -6.56840026e-01  5.79320014e-01 ]
  - 현재는 glove 300차원 pre-trained model을 사용하고 있지만 만약 다른 pre-trained된 모델을 사용하고 싶다면 embedding_dir의 값을 바꾸어주면 된다.
  그리고 text_to_vector의 word_dimension 값을 바꿔주어야 한다. 현재는 300차원이라 300이지만 50차원을 사용하면 50으로 바꿔주어야 한다.
    - Glove pre-trained 다운 받는 곳 : https://nlp.stanford.edu/projects/glove/
    - FASTTEXT pre-trained 다운 받는 곳 : https://fasttext.cc/docs/en/english-vectors.html
    - 그 외 pre-trained 데이터 다운 받는 곳 : https://github.com/RaRe-Technologies/gensim-data 

- build_model
  - 학습 모델 생성
  - binary classification이기 때문에 loss는 binary_crossentropy를 사용했다. output layer에 node를 1로 설정하고 activation은 sigmoid로 설정해 0.5보다 크면 1로 분류했고 0.5이하이면 0으로 분류하였다.

- evaluate : 테스트 데이터를 이용해 모델 성능 평가

## Description
### binary_self_trained.py
- keras embedding layer 사용
- word2vec, fasttext, glove 등의 외부 embedding matrix 값을 사용하지 않고 keras embedding layer를 사용한다.
- binary_pre_trained.py와 다르게 keras embedding layer를 사용하기 때문에 따로 외부에서 embedding matrix 값을 받아오지 않는다.

### binary_elmo.py
- contextualized embedding 사용.
- ELMo는 전체 문장을 넣기 때문에 따로 vector화 시키지 않고 그대로 ELMo에 embedding 입력.
- ELMo에 문장을 입력하면 1024차원으로 압축된 후 dense layer를 거치면 256차원으로 압축.

### multi_pre_trained.py
- pre-trained embedding vector 사용
- multi_train_data의 label은 happy, angry, others, sad로 구성되어 있기 때문에 0, 1, 2, 3으로 바꿔주고 거기서 또 one-hot encoeding으로 변환.
- [0 ,0, 0, 1], [0 ,0, 1, 0], [0 ,1, 0, 0], [1 ,0, 0, 0]로 변경.
- output_layer의 node 개수를 category 개수와 동일하게 설정. (여기서는 4개)
- output_layer의 activate function은 softmax로 설정.
