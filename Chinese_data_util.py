#-*- utf-8 -*-
"""
input: label data, wordvector data
output:
    revs: list.     Store the all informaiton of sentences: label, text, num_words
    vocab: dict: word, nums.    the dict to store all appeared word in train_test data
    w2v: dict: word, word_vector. 
    word_idx_map: dict: word, word_id.
    W: dict: word_id, word_vector. part of vector from Google 300 vector
    W2: dict: word_id, word_vector. All random vector, in range(-0.25, 0.25)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))

"""
import numpy as np
np.random.seed(1337) # for reproducibility

import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import gc
import time


#data_folder: the path to load data
#clean_string : True = clean the sentence
def build_data_cv(data_folder, clean_string=True):
    """
    Loads sentence data ;
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file) as f:
        for line in f:       
            line = line.strip().decode("utf8")
            words = set(line.split(" "))
            '''
            if(len(line.split())> 55):
                continue
            '''
            for word in words:
                vocab[word] += 1
            
            datum  = {"label":1, 
                      "text": line.encode("utf8"),                             
                      "num_words": len(line.split())
                      }
            revs.append(datum)
    with open(neg_file) as f:
        for line in f:       
            line = line.strip().decode("utf8")
            words = set(line.split())
            '''
            if(len(line.split())> 55):
                continue
            '''
            flag = 0
            for word in words:
                vocab[word] += 1
              

            datum  = {"label":0, 
                      "text": line.encode("utf8"),                             
                      "num_words": len(line.split())
                      }

            
            revs.append(datum)

    return revs, vocab
    
def generate_char2vec(alphabet, char2v_dim):
    char2vec = {}
    for alp in range (1,alphabet+1):
        char2vec[alp] = np.random.uniform(-0.25, 0.25, char2v_dim)
    char2vec[" "] = np.random.uniform(-0.25, 0.25, char2v_dim)
    return char2vec

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def char_load_data(path, vector_type="-word2vec", filter_h=5, model_type="CNN"):

    print "Loading data..."
    X = cPickle.load(open(path, "rb"))
    max_len, char2v_dim, revs, char2vec,chinese_character = X[0], X[1], X[2],X[3],X[4]
   

    revs_size = len(revs)
    np.random.seed(1)
    np.random.shuffle(revs)
    pad = filter_h -1
    #gc.collect()
    print "data loaded"
    print "number of sentences: " + str(revs_size)
    print "max sentence length: " + str(max_len)
  

    

    #input_shape
    input_shape = []
    input_shape.append(1)
    input_shape.append(max_len)
    input_shape.append(16)
    input_shape.append(char2v_dim)

    zeros_vector = np.zeros(char2v_dim)
    revs_size=100000
    label = np.empty((revs_size,), dtype="uint8")
    if(model_type == "CNN"):
        data = np.empty((revs_size, input_shape[0], input_shape[1]+2*pad, input_shape[2]+2,input_shape[3]), dtype = "float32")
        for i in range(revs_size):
            rev = revs[i]
            s2v = get_sen_char2v(rev["text"], char2vec,zeros_vector, max_len, 16, input_shape,chinese_character,filter_h)
            data[i,0,:,:,:] = s2v
            label[i] = int(rev["label"])
        label_class = len(set(label))


    return char2v_dim, label_class, data, label

#sen: sentence
#w2v: word to vector
#s2v: sententce to vector
def get_sen_char2v(sent, char2vec, zeros_vector, max_len, max_word_len, input_shape, chinese_character,filter_h=5):
    sent = sent.decode("utf8")
    words = sent.split()
    pad = filter_h - 1
    s2charvec = []
    pad1 = (2*pad+max_len-len(words))/2
    pad2 = 2*pad+max_len-len(words)-pad1

    for i in range(pad1):
        temp=[]
        temp.append(char2vec[" "])
        for j in range(max_word_len):
            temp.append( zeros_vector )
        temp.append(char2vec[" "])
        s2charvec.append(temp)

    for word in words:
        temp=[]
        temp.append(char2vec[" "])
        gap = int( (max_word_len - len(word*4)) /2)
        for i in range(gap):
            temp.append(zeros_vector)
        for alp in word:
            chinese_tmp=chinese_character[alp]
            now = min(4,len(chinese_tmp))
            for i in range(now):
                temp.append(char2vec[chinese_tmp[i]])
            for i in range(0,4-now):
                temp.append(zeros_vector)
            
        for i in range(max_word_len-len(word*4)-gap):
            temp.append(zeros_vector)
        temp.append(char2vec[" "])
        s2charvec.append(temp)

    for i in range(pad2):
        temp=[]
        temp.append(char2vec[" "])
        for j in range(max_word_len):
            temp.append( zeros_vector )
        temp.append(char2vec[" "])
        s2charvec.append(temp)

       
    return s2charvec








if __name__=="__main__":    

    #config
    data_folder = ["./clean_chinese_positive","./clean_chinese_negative"]    
    char2v_dim = 15

    chinese_character=dict()

    with open("./clean_chinese_character_encode") as f:
        for line in f.readlines():
            k=line.strip().split("\t")
            k[0]=k[0].decode("utf8")
            print (k[0])
            print (k[1])
            tmp=k[1].split(" ")
            chinese_character[k[0]]=[int(x) for x in tmp]

    char2vec = generate_char2vec(215, char2v_dim)


    print "loading data...",        
    revs, vocab= build_data_cv(data_folder, clean_string=True)


    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "generate char2vec vectors...",
    
    #w2v = load_bin_vec(w2v_file, vocab)
    print "char2vec generate!"

    cPickle.dump([max_l,char2v_dim, revs,char2vec,chinese_character],open("chinese_charmr_big", "wb"))
    print "dataset created!"
    
