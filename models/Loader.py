"""
The main class to load all the predictions with different trained algorithms. One instance of it is automatically created in the initializing of the package and its called "Predictor". 
Error-site codes and log files should be passed to it and a list of Prediction instances will be returned.
:author: Hamed Bakhshiansohi <hbakhshi@cern.ch>
"""

from . import SiteErrorCodeModelLoader as SECML
from AIErrorHandling.training.SitesErrorCodes import SitesErrorCodes_path 

class Loader :
    def __init__(self):
        """
        new models should be added manually to the list of AllModels.
        """
        self.AllModels = []
        self.AllModels.append( SECML.SiteErrorCodeModelLoader( SitesErrorCodes_path+"/models/FirstTry" ) )
        

    def __call__(self , **inputs):
        """
        gets all the information about the failed job and returns a list of Predictions
        :param kwargs inputs: all the possible inputs from the console
        :param dict good_sites: map of good sites and number of failed jobs in eash site, like what is provided by 'actionhistory' in old-console
        :param dict bad_sites: map of bad sites and number of failed jobs in eash site, like what is provided by 'actionhistory' in old-console
        """
        ret = [model(**inputs) for model in self.AllModels]
        return ret
