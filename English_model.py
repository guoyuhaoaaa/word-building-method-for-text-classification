import numpy as np
import sys
import time

#np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.callbacks import Callback
from keras.optimizers import Adadelta, SGD, RMSprop, Adagrad, Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, TimeDistributedDense, Masking, Reshape, Highway,Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Convolution3D, MaxPooling3D
from keras.layers.recurrent import LSTM, GRU
from keras.utils.np_utils import accuracy
from keras.utils import np_utils, generic_utils
from guo_char_process_data import char_load_data
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization 

def DNN(data, label_class):
    input_shape=[]
    input_shape.append(data.shape[1])
    input_shape.append(data.shape[2])
    input_shape.append(data.shape[3])
    input_shape.append(data.shape[4])
    
    print (input_shape)

   
    filter_hs = [100,100,100]
    convolutional_di=[3,5,4]
    
    sub_models = []
   
    print("Begin build model..")
    for i in range(0,3):
        sub_model = Sequential()
        sub_model.add(Convolution3D( nb_filter = filter_hs[i], kernel_dim1=1,
            kernel_dim2=convolutional_di[i]
            ,kernel_dim3=input_shape[3],border_mode = 'valid', activation =
            'relu', 
            input_shape=(input_shape[0],input_shape[1],input_shape[2],input_shape[3])
            ))
        #sub_model.add(BatchNormalization())
        sub_model.add(Permute((2,1,3,4)))
        sub_model.add( MaxPooling3D(pool_size=(1,input_shape[2]-convolutional_di[i]+1,1)) )
        sub_model.add(Reshape((input_shape[1],filter_hs[i])))
        sub_models.append(sub_model)

    model = Sequential()
    model.add(Merge(sub_models, mode='concat'))
    #model.add(BatchNormalization())
    model.add( LSTM(185))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(output_dim = label_class, activation='softmax'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer = 'rmsprop')

    return model

if __name__ == "__main__":
    if(sys.argv[1] == ""):
        print ("argv[1] is the path of file made from guo_process_data.py")
        exit()
    #config
    seed = np.random.randint(0,1000)
    print (seed)
    #seed = 1337
    batch_size = 16
    nb_epoch = 25
    all_num = 1
    valid_rate = 0.9

    char2v_dim, label_class, data, label = char_load_data(path
            =sys.argv[1],filter_h =5, model_type = "CNN");
    #label need to change
    label = np_utils.to_categorical(label, label_class)

    acces = np.zeros(all_num)
    for i in range(0, all_num):
        #random data
        np.random.seed(seed) 
        np.random.shuffle(data)
        np.random.seed(seed) 
        np.random.shuffle(label)

        #data split to train and test
        datasize = len(label)
        train_data = data[ : int(datasize*valid_rate)]
        train_label = label[ : int(datasize*valid_rate)]
        #valid_data = data[ datasize *valid_rate: datasize*(valid_rate+(1-valid_rate)/2)]
        #valid_label = label[ datasize *valid_rate: datasize*(valid_rate+(1-valid_rate)/2)]
        test_data = data[int(datasize*valid_rate): ]
        test_label = label[int(datasize*valid_rate):]

        print ("trainNum: %d"%(i))
        print ("train_file: %d"%(len(train_label)))
        print ("test_file: %d"%(len(test_label)))

        model = DNN(train_data,  label_class)

        print("Begin train model..")
        hist = model.fit([train_data, train_data, train_data], train_label, 
            batch_size = batch_size, 
            nb_epoch = nb_epoch,
            shuffle = True,
            validation_data=([test_data,test_data,test_data],test_label),
            verbose = 1
            )
