"""
a class to load a model that has been trained using the table of sites and errors
:author: Hamed Bakhshiansohi <hbakhshi@cern.ch>
"""

import numpy as np
import json
import tensorflow.keras as keras
from AIErrorHandling.training.SitesErrorCodes.Tasks import Task
from . import Prediction

class SiteErrorCodeModelLoader :
    def __init__(self , model_file_name ):
        """
        loads the trained model
        :param str model_file_name: two files are expected to exist in this address : one json and one hdf5 file. They should be stored using the SaveModel method of the DNNTrain class
        """
        self.model = keras.models.load_model( model_file_name + ".hdf5" )
        JSON = json.load( open(model_file_name + ".json" ) )
        self.all_sites = list(JSON['all_sites'])
        self.all_errors = list(JSON['all_errors'])
        self.all_actions = list(JSON['all_actions'])
        self.IsBinary = bool(JSON['IsBinary'])
        self.TiersOnly = bool(JSON['TiersOnly'])
        self.Task = Task({} , "TaskLoader" , self)

        self.ModelID = int( JSON['model'] )
        self.InputTrainingDataID = int( JSON['trainingdata'])

        self.Prediction = Prediction.Prediction( self.ModelID , self.InputTrainingDataID ) 
        
    def __call__(self , good_sites , bad_sites ):
        """
        returns the prediction of the model for the input good/bad site errors
        :param dict good_sites: map of good sites and number of failed jobs in eash site, like what is provided by 'actionhistory' in old-console
        :param dict bad_sites: map of bad sites and number of failed jobs in eash site, like what is provided by 'actionhistory' in old-console
        """
        self.Task.normalize_errors( good_sites , bad_sites , self.TiersOnly )
        prediction = self.model.predict( np.array( [ self.Task.Get2DArrayOfErrors() ] ) )
        PRED = str(prediction)
        if self.IsBinary:
            PRED = self.all_actions[ prediction[0][0] > 0.5 ]
        self.Prediction.SetValues( PRED , "" , "" )
        return self.Prediction

