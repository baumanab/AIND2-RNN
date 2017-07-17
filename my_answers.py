import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):

    # containers for input/output pairs
    X = []
    y = []
    
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        
    y= series[window_size:]    

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    
    model = Sequential()
    model.add(LSTM(5,  input_shape = (window_size,1)))
    model.add(Dense(1))
    
    return model

    


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    
    import re
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    pattern = re.compile('[^a-zA-Z!.?:;,\s]') # match by negate
    text= pattern.sub('', text)

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    
    # containers for input/output pairs  
    X = []
    y = []
    
    i= 0
    text_length= len(text)
    while i + window_size < text_length: # while window upper bound is < entire string
        X.append(text[i:i + window_size])
        y.append(text[i + window_size])
        i += step_size
        
    return X,y

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    
    from keras.layers import Dense, Activation, LSTM
    
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars))) # shape is windowsize, unique characters
    model.add(Dense(num_chars)) # fully connect layer with as many outputs as uniqe characters
    model.add(Activation('softmax')) # softmax for multi-class
    
    return model