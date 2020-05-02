# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:23:29 2018

@author: Kuo
"""
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM

import matplotlib.pyplot as plt  
from keras.utils.vis_utils import plot_model

def show_train_history(train_history, train, model, tp):
    plt.figure()
    plt.plot(train_history.history[train], label=train )
    plt.plot(train_history.history['val_'+train], label='val_' + train)
    plt.title(model+"("+tp+")"+" Train History")
    plt.xlabel("Epoch")
    plt.ylabel(train)
    plt.legend(loc="upper left")
    plt.savefig('./'+model+'_'+train+'('+tp+')'+'.png')
    plt.show()
    

def readFile(filename, sp):
    data = pd.read_csv(filename, sep='\n', header=None)
    data = np.array(data.loc[:,0].str.split(sp).tolist())
    return data[:,-1], data[:,0]

if __name__=='__main__':
    train_x, train_y = readFile('./training_label.txt', '\+\+\+\$\+\+\+')
    test_x, test_y = readFile('./testing_label.txt', r'\#\#\#\#\#')
    
    token = Tokenizer(num_words = 3800)
    token.fit_on_texts(train_x)
    
    train_x_seq = token.texts_to_sequences(train_x)
    test_x_seq = token.texts_to_sequences(test_x)
    
    train_x = sequence.pad_sequences(train_x_seq, maxlen=100)
    test_x = sequence.pad_sequences(test_x_seq, maxlen=100)
    
# =============================================================================
#     RNN
# =============================================================================
    #Without dropout
    modelRNN = Sequential()  
    modelRNN.add(Embedding(output_dim=32, input_dim=3800, input_length=100)) 
    modelRNN.add(SimpleRNN(units=16))
    modelRNN.add(Dense(units=128,activation='relu'))
    modelRNN.add(Dense(units=256,activation='relu'))
    modelRNN.add(Dense(units=1,activation='sigmoid'))
    modelRNN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
    plot_model(modelRNN, to_file='RNN_noDropout.png', \
               show_shapes=True, show_layer_names=True)
    
    RNN_train_history = modelRNN.fit(train_x, train_y, 
         epochs = 15, batch_size=256, verbose = 0, validation_split=0.2)
    
    RNN_scores = modelRNN.evaluate(test_x, test_y, verbose=1)
    print('[no Dropout] RNN testing acc = ', RNN_scores[1])
    
    #dropout
    modelRNN2 = Sequential()  
    modelRNN2.add(Embedding(output_dim=32, input_dim=3800, input_length=100)) 
    modelRNN2.add(Dropout(0.7)) 
    modelRNN2.add(SimpleRNN(units=16))
    modelRNN2.add(Dense(units=128,activation='relu'))
    modelRNN2.add(Dropout(0.7))
    modelRNN2.add(Dense(units=256,activation='relu'))
    modelRNN2.add(Dense(units=1,activation='sigmoid'))
    modelRNN2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
    plot_model(modelRNN2, to_file='RNN_Dropout.png', \
               show_shapes=True, show_layer_names=True)
    
    RNN_train_history2 = modelRNN2.fit(train_x, train_y, 
         epochs = 15, batch_size=256, verbose = 0, validation_split=0.2)
    RNN_scores2 = modelRNN2.evaluate(test_x, test_y, verbose=0)
    print('[Dropout] RNN testing acc = ', RNN_scores2[1])
    
    show_train_history(RNN_train_history, 'acc', 'RNN', 'without dropout')
    show_train_history(RNN_train_history, 'loss', 'RNN', 'without dropout')
    
    show_train_history(RNN_train_history2, 'acc', 'RNN', 'dropout')
    show_train_history(RNN_train_history2, 'loss', 'RNN', 'dropout') 
    
    
# =============================================================================
#     LSTM
# =============================================================================
    #Without dropout
    modelLSTM = Sequential() 
    modelLSTM.add(Embedding(output_dim=32, input_dim=3800, input_length=100))
    
    modelLSTM.add(LSTM(32))
    modelLSTM.add(Dense(units=128,activation='relu'))
    modelLSTM.add(Dense(units=256,activation='relu'))
    modelLSTM.add(Dense(units=1,activation='sigmoid'))
    modelLSTM.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
    plot_model(modelLSTM, to_file='LSTM_noDropout.png', \
               show_shapes=True, show_layer_names=True)
    
    LSTM_train_history = modelLSTM.fit(train_x, train_y, epochs = 15, \
                                       batch_size=256, verbose = 0, validation_split=0.2)
    LSTM_scores = modelLSTM.evaluate(test_x, test_y, verbose=0)
    print('[no Dropout] LSTM testing acc = ', LSTM_scores[1])
    
    #dropout
    modelLSTM2 = Sequential() 
    modelLSTM2.add(Embedding(output_dim=32, input_dim=3800, input_length=100))
    modelLSTM2.add(Dropout(0.7))
    modelLSTM2.add(LSTM(32))
    modelLSTM2.add(Dense(units=128,activation='relu'))
    modelLSTM2.add(Dropout(0.7))
    modelLSTM2.add(Dense(units=256,activation='relu'))
    modelLSTM2.add(Dense(units=1,activation='sigmoid'))
    modelLSTM2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
    plot_model(modelLSTM2, to_file='LSTM_Dropout.png', \
               show_shapes=True, show_layer_names=True)

    LSTM_train_history2 = modelLSTM2.fit(train_x, train_y, epochs = 15, \
                                         batch_size=256, verbose = 0, validation_split=0.2)
    LSTM_scores2 = modelLSTM2.evaluate(test_x, test_y, verbose=0)
    print('[Dropout] LSTM testing acc = ', LSTM_scores2[1])
    
    show_train_history(LSTM_train_history, 'acc', 'LSTM', 'without dropout')
    show_train_history(LSTM_train_history, 'loss', 'LSTM', 'without dropout')
    
    show_train_history(LSTM_train_history2, 'acc', 'LSTM', 'dropout')
    show_train_history(LSTM_train_history2, 'loss', 'LSTM', 'dropout')
    

          
                                
                                
    
    