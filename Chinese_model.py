import numpy as np
import sys
import time

#np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.callbacks import Callback
from keras.optimizers import Adadelta, SGD, RMSprop, Adagrad, Adam
from keras.models import Sequential,Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, TimeDistributedDense, Masking, Reshape, Highway,Permute
from keras.layers.convolutional import Convolution1D, MaxPooling1D,Convolution3D, MaxPooling3D
from keras.layers.recurrent import LSTM, GRU
from keras.utils.np_utils import accuracy
from keras.utils import np_utils, generic_utils
from char_process_data import char_load_data
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization 

def DNN(data, data1 , label_class):
    input_shape=[]
    input_shape.append(data.shape[1])
    input_shape.append(data.shape[2])
    input_shape.append(data.shape[3])
    input_shape.append(data.shape[4])
    
    print (input_shape)

   
    filter_hs = [100,100,100]
    convolutional_di=[8,8,8]
    
   
    print("Begin build model..")

    model = Graph()
    model.add_input(name='input', input_shape=(input_shape[0], input_shape[1], input_shape[2],input_shape[3]), dtype='float32')

    model.add_input(name='input1', input_shape=(data1.shape[1],data1.shape[2] ), dtype='float32')

    #model.add_node(Reshape((input_shape[0]*input_shape[1]*input_shape[2],input_shape[3])),name='rrr',input='input')
    #model.add_node(Convolution1D(input_shape[3],1,activation='relu'),name='cnn',input='rrr')
    #model.add_node(Reshape((input_shape[0],input_shape[1],input_shape[2],input_shape[3])),name='ttt',input='cnn')
    '''
    i=0
    model.add_node(Convolution3D( nb_filter = filter_hs[i], kernel_dim1=1,
            kernel_dim2=convolutional_di[i]
            ,kernel_dim3=input_shape[3],border_mode = 'valid',
            activation='relu',dim_ordering='th'),name='cnn1',input='input')
    model.add_node(Permute((2,1,3,4)),name='pe1',input='cnn1')
    model.add_node(MaxPooling3D(pool_size=(1,input_shape[2]-convolutional_di[i]+1,1),dim_ordering='th') ,name='max1',input='pe1')
    model.add_node(Reshape((input_shape[1],filter_hs[i])),name='re1',input='max1')

    i=1
    model.add_node(Convolution3D( nb_filter = filter_hs[i], kernel_dim1=1,
            kernel_dim2=convolutional_di[i]
            ,kernel_dim3=input_shape[3],border_mode = 'valid', activation
            ='relu',dim_ordering='th'),name='cnn2',input='input')
    model.add_node(Permute((2,1,3,4)),name='pe2',input='cnn2')
    model.add_node(MaxPooling3D(pool_size=(1,input_shape[2]-convolutional_di[i]+1,1),dim_ordering='th') ,name='max2',input='pe2')
    model.add_node(Reshape((input_shape[1],filter_hs[i])),name='re2',input='max2')

    i=2
    model.add_node(Convolution3D( nb_filter = filter_hs[i], kernel_dim1=1,
            kernel_dim2=convolutional_di[i]
            ,kernel_dim3=input_shape[3],border_mode = 'valid', activation
            ='relu',dim_ordering='th'),name='cnn3',input='input')
    model.add_node(Permute((2,1,3,4)),name='pe3',input='cnn3')
    model.add_node(MaxPooling3D(pool_size=(1,input_shape[2]-convolutional_di[i]+1,1),dim_ordering='th') ,name='max3',input='pe3')
    model.add_node(Reshape((input_shape[1],filter_hs[i])),name='re3',input='max3')

    '''
   
    
    #model.add_node(LSTM(200),name='lstm',inputs=['re1','re2','re3'],concat_axis=-1,merge_mode='concat')
    model.add_node(LSTM(200),name='lstm',inputs=['input1','input1'],concat_axis=-1,merge_mode='concat')
    #model.add_node(LSTM(200),name='lstm',input='input1')




    model.add_node(Dropout(0.5), name='drop', input='lstm')
    model.add_node(Dense(label_class),name='fullconnection', input='drop')
    model.add_node(Activation('softmax'), name='softmax', input='fullconnection')
    model.add_output(name='output', input='softmax')
    model.compile(loss={'output': 'categorical_crossentropy'},optimizer='rmsprop', metrics=["accuracy"])

    return model

if __name__ == "__main__":
    if(sys.argv[1] == ""):
        print ("argv[1] is the path of file made from guo_process_data.py")
        exit()
    #config
    seed = np.random.randint(0,1000)
    print (seed)
    batch_size = 200
    nb_epoch = 205
    all_num = 1
    valid_rate = 0.9

    w2v_dim,char2v_dim, label_class, data, data1,label = char_load_data(path
            =sys.argv[1],filter_h =5, model_type = "CNN");
    #label need to change
    label = np_utils.to_categorical(label, label_class)

    acces = np.zeros(all_num)
    for i in range(0, all_num):

        datasize = len(label)
        train_data = data[ : int(datasize*valid_rate)]
        train_data1 = data1[ : int(datasize*valid_rate)]
        train_label = label[ : int(datasize*valid_rate)]
        test_data = data[int(datasize*valid_rate): ]
        test_data1 = data1[int(datasize*valid_rate): ]
        test_label = label[int(datasize*valid_rate):]

        print ("trainNum: %d"%(i))
        print ("train_file: %d"%(len(train_label)))
        print ("test_file: %d"%(len(test_label)))

        model = DNN(train_data, train_data1, label_class)

        print("Begin train model..")
        hist = model.fit(
                {'input': train_data, 'input1':train_data1,'output': train_label},
            batch_size = batch_size, 
            nb_epoch = nb_epoch,
            shuffle = True,
            validation_data={'input': test_data,'input1':test_data1, 'output': test_label},
            verbose = 1
            )

