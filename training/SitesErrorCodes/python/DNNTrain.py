import CMS_AIErrorHandling as AIErrHand

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD,Adam
from keras import regularizers
from keras import metrics
from sklearn.metrics import roc_curve,auc

random_seed = AIErrHand.random_seed

def top_first_categorical_accuracy(kk , name):
    def ktop(x , y ):
        return metrics.top_k_categorical_accuracy(x, y, kk)

    ktop.__name__ = name
    return ktop

class DNNTrain :
    def __init__(self , tasks , train_ratio=0.8 , model = None ):
        self.Tasks = tasks
        self.X_train, self.y_train, self.X_test, self.y_test = tasks.GetTrainTestDS( train_ratio )

        self.X_train = self.X_train.astype('int16')
        self.X_test = self.X_test.astype('int16')

        if self.Tasks.IsBinary :
            self.Y_train = self.y_train
            self.Y_test = self.y_test
        else:
            self.Y_train = np_utils.to_categorical(self.y_train, len(tasks.all_actions) , 'int8')
            self.Y_test = np_utils.to_categorical(self.y_test, len(tasks.all_actions) , 'int8')

        self.model = model
        

    def MakeModel(self, flatten=True , layers=[] , optimizer='adam' , loss='categorical_crossentropy' ):
        self.model = Sequential()
        if flatten :
            self.model.add(Flatten())

        for layer in layers :
            nNeurons = layer[0]
            regularizer = layer[1]
            activation = layer[2]

            self.model.add(Dense(nNeurons,
                                 kernel_regularizer= regularizers.l2(regularizer) if regularizer else None ,
                                 kernel_initializer=keras.initializers.RandomNormal(seed=random_seed),
                                 bias_initializer=keras.initializers.RandomNormal(seed=random_seed*2),
                                 activation=activation ) )

        if self.Tasks.IsBinary :
            self.model.add( Dense( 1 , activation='sigmoid' , kernel_initializer=keras.initializers.RandomNormal(seed=random_seed*3) , bias_initializer=keras.initializers.RandomNormal(seed=random_seed*4) ) )
        else:
            self.model.add( Dense( len(self.Tasks.all_actions) ,
                                   activation='softmax' ,
                                   kernel_initializer=keras.initializers.RandomNormal(seed=random_seed*5),
                                   bias_initializer=keras.initializers.RandomNormal(seed=random_seed*6) ) )

            
        if optimizer == "sgd" :
            Optimizer = SGD(lr=.5)
        elif optimizer == "adam":
            Optimizer =  Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else :
            Optimizer = optimizer

        self.model.compile(
            loss=loss ,#'categorical_crossentropy', 'mean_squared_error', 'categorical_crossentropy' 'mean_absolute_error'
            optimizer=Optimizer,
            metrics=['accuracy']
            # , 'categorical_accuracy' , top_first_categorical_accuracy(1,"kfirst"), top_first_categorical_accuracy(2,"kfirsttwo"),top_first_categorical_accuracy(3,"kfirstthree")]
        )

    def Fit(self,batch_size=100, epochs=10 , validation_split=0.0 , verbose=1):
        return self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose , validation_split=validation_split )

    def ROC(self):
        self.roc_fpr, self.roc_tpr, self.roc_thresholds = roc_curve(self.Y_test, self.y_prediction.ravel() )
        self.roc_auc = auc(self.roc_fpr , self.roc_tpr)

        self.roc_plot = plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(self.roc_fpr, self.roc_tpr, label='Keras (area = {:.3f})'.format(self.roc_auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
        
    def Test(self):
        self.y_prediction = self.model.predict(self.X_test) #, self.Y_test, verbose=verbose )
        self.ROC()
        results = zip(self.y_prediction, self.Y_test)

        if not self.Tasks.IsBinary :
            average_per_true = np.zeros( [len(self.Tasks.all_actions)+1, len(self.Tasks.all_actions)] )
            for pre,true in results:
                index = list( true ).index(1)
                for i in range (0, len(average_per_true[ index ]) ):
                    average_per_true[index][i] += pre[i]
                average_per_true[-1][ index ] += 1

            for iii in range(0,len(self.Tasks.all_actions) ) :
                row = average_per_true[iii]
                total = average_per_true[-1][iii]
                if total != 0 :
                    row /= total

            print(average_per_true)
        return results
