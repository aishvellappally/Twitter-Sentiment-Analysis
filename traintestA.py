import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D
from tensorflow.keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

import re
import tensorflow.keras.callbacks
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
import nltk
from nltk.corpus import stopwords
import argparse


def binarize(x, sz=72):
    return tf.cast(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1),dtype=tf.float32)

def binarize_outshape(in_shape, sz=72):
    return in_shape[0], in_shape[1], sz


def striphtml(s):
    p = re.compile(r'<.*?>')
    return p.sub('', s)


def clean(s):
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', s) # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    #tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
    return tweet

BATCH_SIZE = 32
epochs = 50

parser = argparse.ArgumentParser(description='Load data and path to model')
parser.add_argument('--in_1', type=str, help='relative path to data1')
parser.add_argument('--in_2', type=str, help='relative path to data2')
parser.add_argument('--in_3', type=str, help='relative path to data3')
parser.add_argument('--in_4', type=str, help='relative path to data4')
parser.add_argument('--in_5', type=str, help='relative path to data5')
parser.add_argument('--in_6', type=str, help='relative path to data6')
parser.add_argument('--model_path', type=str, help='path to model')



args = vars(parser.parse_args())



df_train_2015 = pd.read_excel(args['in_1'], names=('id', 'topic', 'label','tweet'), header=None)
df_test_2015 = pd.read_excel(args['in_2'], names=('id', 'topic', 'label','tweet'),header=None)
df_train_2016=pd.read_excel(args['in_3'], names=('id', 'topic', 'label','tweet'),header=None)
df_dev = pd.read_excel(args['in_4'], names=('id', 'topic', 'label','tweet'),header=None)
df_devtest = pd.read_excel(args['in_5'], names=('id', 'topic', 'label','tweet'),header=None)
df_test_2016 = pd.read_excel(args['in_6'], names=('id', 'topic', 'label','tweet'),header=None)

df = pd.concat([df_train_2015,df_test_2015,df_train_2016,df_test_2016,df_dev,df_devtest])
labels = df['label'].reset_index(drop=True) # label dataframe


df.replace({'label': {'positive': 1, 'negative': -1, 'neutral':0}},inplace=True)
# CNN with entire tweet and topic #Character

def pre_process(df):
  df['combined']=df['tweet']+' '+df['topic']
  txt = ''
  docs = []
  sentences = []
  sentiments = []
  tweets=[]
  i=0
  stop_words = set(stopwords.words('english') + list(punctuation) + ['at_user','url'])

  '''for cont, sentiment in zip(df_train.combined, df_train.label):
      #sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean(striphtml(cont)))
      cont.lower()
      tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', cont) # remove URLs
      tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
      tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
      tweet = word_tokenize(tweet)
      docs.append(tweet)
      sentiments.append(sentiment)'''
  for cont, sentiment in zip(df.combined, df.label):
      strip=striphtml(cont)
      cleaned=clean(strip)
      sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', cleaned)
      sentences = [sent.lower() for sent in sentences]
      sentences = [word_tokenize(sent) for sent in sentences]
      sentence_fin =[]
      for sent in sentences:
          sentence_fin.append([word for word in sent if word not in stop_words])
      docs.append(sentence_fin)
      sentiments.append(sentiment)

  print(docs[80:90])
  #find max number of sentences.
  i=0
  max_i=0
  for sent in docs:
      i=0
      for k in sent:
          i=i+1
      max_i=max(i,max_i)
      if(i==max_i):
          j_max=sent

  #find max number of characters in a tweet
  max_char=0
  for doc in docs:
      count=0
      for sent in doc:
          for word in sent:
              for s in word:
                  count+=1
                  max_char=max(max_char,count)
  print(max_char)
  for doc in docs:
      for sent in doc:
          for word in sent:
              for s in word:
                  txt += s

  chars = set(txt)
  print('total chars:', len(chars))
  char_indices = dict((c, i) for i, c in enumerate(chars))
  indices_char = dict((i, c) for i, c in enumerate(chars))

  maxlen = 140
  max_sentences = 10

  X = np.ones((len(docs), max_sentences, maxlen), dtype=np.int64) * -1
  y = np.array(sentiments)
  max_j=0
  for i, doc in enumerate(docs):
      for j, sentence in enumerate(doc):
          if j < max_sentences:
              for word in sentence[-maxlen:]:
                  for t, char in enumerate(word):
                      X[i, j, (maxlen-1-t)] = char_indices[char]

  #ONE HOT ENCODING LABELS
  #label_encoder = LabelEncoder()
  #integer_category = label_encoder.fit_transform(df.label)
  #y = sentiments
  #shuffle X and y
  ids = np.arange(len(X))
  np.random.shuffle(ids)

  # shuffle
  X = X[ids]
  y = y[ids]

  print("Shape of X ", X.shape)
  print("Shape of Y ", y.shape)
  return X,y,chars, max_sentences, maxlen

X, y,chars, max_sentences, maxlen = pre_process(df)
num_classes = 3

def onehot(arr, num_class):
    return np.eye(num_class)[np.array(arr.astype(int)).reshape(-1)]

y = onehot(y, num_classes)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
X_train,X_val,y_train,y_val = train_test_split(X_train , y_train , test_size=0.25, random_state = 42)

'''def create_model(chars,max_sentences,maxlen):
  filter_length = [5, 3]
  nb_filter = [8, 16
  pool_length = 2
  # document input
  document = Input(shape=(max_sentences, maxlen), dtype='int64')
  # sentence input
  in_sentence = Input(shape=(maxlen,), dtype='int64')
  # char indices to one hot matrix, 1D sequence to 2D
  embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
  # embedded: encodes sentence
  for i in range(len(nb_filter)):
      embedded = Conv1D(filters=nb_filter[i],
                        kernel_size=filter_length[i],
                        padding='same',
                        activation='relu',
                        kernel_initializer='he_normal',
                        strides=1)(embedded)

      embedded = MaxPooling1D(pool_size=pool_length)(embedded)
      embedded = Dropout(0.3)(embedded)


  bi_lstm_sent = Bidirectional(LSTM(25, return_sequences=False, dropout=0.3, recurrent_dropout=0.3, implementation=0))(embedded)

  # sent_encode = merge([forward_sent, backward_sent], mode='concat', concat_axis=-1)
  sent_encode = Dropout(0.3)(bi_lstm_sent)
  # sentence encoder
  encoder = Model(inputs=in_sentence, outputs=sent_encode)
  encoder.summary()

  encoded = TimeDistributed(encoder)(document)
  # encoded: sentences to bi-lstm for document encoding
  b_lstm_doc = Bidirectional(LSTM(5, return_sequences=False, dropout=0.3, recurrent_dropout=0.3, implementation=0))(encoded)

  output = Dropout(0.3)(b_lstm_doc)
  output = Dense(5, activation='relu')(output)
  output = Dropout(0.3)(output)
  output = Dense(1, activation='sigmoid')(output)

  model = Model(inputs=document, outputs=output)

  model.summary()
  model.compile(loss='cateogorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

model = create_model(chars,max_sentences,maxlen)
checkpoint = ModelCheckpoint("model_semeval_taskA.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
callback = [checkpoint,early]

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = epochs, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),verbose = 1, callbacks = callback)
'''
modelPath = args['model_path']
model_test = load_model(modelPath)
predictions = model_test.predict(X_test, batch_size = BATCH_SIZE, verbose = 1)

pred = np.zeros((len(y_test),3))
for i in range(len(y_test)):
  count = 0
  for val in predictions[i]:
    if val == max(predictions[i]):
      pred[i][count]=1
    else:
      pred[i][count]=0
    count=count+1

test_labels = np.zeros((len(y_test)))
for i in range(len(y_test)):
  if y_test[i][0] == 1:
    test_labels[i]=0
  elif y_test[i][1] ==1:
    test_labels[i]=1
  elif y_test[i][2] ==1:
    test_labels[i]=-1

pred_labels = np.zeros((len(y_test)))
for i in range(len(y_test)):
  if pred[i][0] == 1:
    pred_labels[i]=0
  elif pred[i][1] ==1:
    pred_labels[i]=1
  elif pred[i][2] ==1:
    pred_labels[i]=-1

#compare with baselines
df_base = pd.DataFrame(test_labels, columns=['True'])
df_base['all positive'] = 1.0
df_base['all negative'] = -1.0
df_base['all neutral'] = 0.0
df_base['predicted']= pred_labels
df_base=df_base.reset_index(drop=True)
Acc_1=accuracy_score(df_base['True'],df_base['all positive'])
Acc_2=accuracy_score(df_base['True'],df_base['all negative'])
Acc_3=accuracy_score(df_base['True'],df_base['all neutral'])
Acc_4=accuracy_score(df_base['True'],df_base['predicted'])
print("Accuracies Base1, Base2, Base3, Model: ", Acc_1, Acc_2 , Acc_3, Acc_4)
