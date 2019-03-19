"""
DNN training module to train on the action history
:author: Hamed Bakhshiansohi <hbakhshi@cern.ch>
"""

from . import random_seed, SitesErrorCodes_path
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

def top_first_categorical_accuracy(kk , name):
    """
    a method to create methods to be used as a metric.
    :param int kk: the accuracy in the first kk categories will be checked
    :param str name: the name of the metric
    """
    def ktop(x , y ):
        return metrics.top_k_categorical_accuracy(x, y, kk)

    ktop.__name__ = name
    return ktop

class DNNTrain :
    def __init__(self , tasks , train_ratio=0.8):
        """
        :param Tasks tasks: an instance of Tasks class
        :param float train_ratio: a number beween 0 and 1, specifying the ratio of data that is to be used for training
        """
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


    def MakeModel(self, flatten=True , layers=[] , optimizer='adam' , loss='categorical_crossentropy' ):
        """
        to make the model and compile it, if the input are binary a layer with sigmoid activation is added at the end. otherwise, a layer with softmax is inserted
        :param bool flatten: by default for the Task object it should be true
        :param list layers: list of layer, each item should be of the format of (nNeurons, regularizer, activation). if regularizer is None, no regularization is done at this layer
        :param optimizer: name of the optimizer, or an instance of the optimizer to be used
        :param str loss: name of the loss function
        """
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
        """
        do the fit of training. standard parameters of keras.Model.fit
        """
        return self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose , validation_split=validation_split )

    def ROC(self):
        """
        plot ROC curve for test dataset
        """
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
        
    def Test(self , plot_roc = False):
        """
        run the test and returns a map of predictions and true values.
        """
        self.y_prediction = self.model.predict(self.X_test) #, self.Y_test, verbose=verbose )
        if plot_roc:
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

    def SaveModel(self, file_name , model_details , trainingdata_details):
        """
        Save the model to two files :
        one hdf5 file and one json file under the models subdirectory of the current package are created with file_name
        :param str file_name: the name of the file, without any extra extension
        :param int model_details: the integer id of the model. this value will be stored in json file for future references. authors and developers should keep track of its values to make sense.
        :param int trainingdata_details: the integer id of the dataset that was used for training the model. this value will be stored in json file for future references. authors and developers should keep track of its values to make sense.
        """
        self.model.save(SitesErrorCodes_path + "/models/" + file_name + ".hdf5")
        with open( SitesErrorCodes_path + "/models/" + file_name + '.json', 'w') as fp:
            json.dump({'all_sites':self.Tasks.all_sites ,
                       'all_errors':self.Tasks.all_errors ,
                       'all_actions':self.Tasks.all_actions ,
                       'TiersOnly':self.Tasks.TiersOnly ,
                       'IsBinary':self.Tasks.IsBinary ,
                       'model':model_details,
                       'trainingdata':trainingdata_details} , fp)
