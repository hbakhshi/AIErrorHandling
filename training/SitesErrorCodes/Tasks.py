"""
Thid module contains classes to read contents of tasks and actions from the 'actionhistory' json file, produced by old-console
Here is the list of classes : 
1. Task : it gets one entry from json file and interpret it. It converts errors-sites to numpy array
2. Task.Action : details of the action taken by the operator 
3. Tasks : list of tasks. It includes lists of all sites, all errors and all actions in the json file. 
One can ask Tasks class to categorize tasks based on the site-tier instead of site-name. It is also capable of converting the information to 'acdc/non-acdc' binary decision.

:author: Hamed Bakhshiansohi <hbakhshi@cern.ch>
"""

import workflowwebtools as wwt
from pandas import DataFrame

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras


class Task:
    """
    class to capsulate information about each task and the action took by the operator
    """
    class Action:
        """
        class to store Action taken by operator
        """
        def __init__(self,parameters, all_actions):
            """
            :param dict parameters: information from json file for the action
            :param list all_actions: list of the name of all the actions, so the index of the current action can be found
            """
            self.sites = parameters.get("sites")
            self.splitting  = parameters.get('splitting')
            self.xrootd  = parameters.get('xrootd')
            self.other  = parameters.get('other')
            self.memory  = parameters.get('memory')
            self.action  = str( parameters.get('action') )
            self.cores  = parameters.get('cores')
            self.secondary  = parameters.get('secondary')

            self.SetCode( all_actions )

        def SetCode(self , all_actions , non_acdc_index = 0):
            """
            set the action code. here the action code is set to the corresponding index of non-acdc if the action is not found in the list of all_actions
            :param list all_actions: list of all actions. 
            :param int non_acdc_index: by default it assumes that 'non-acdc' is the first element in the list, but it can be modified by this parameter
            """
            try :
                self.action_code = all_actions.index( self.action )
            except ValueError as err :
                if all_actions[ non_acdc_index ] == "non-acdc" :
                    self.action_code = non_acdc_index
                else :
                    raise err
            
        def code(self):
            """
            return the action code
            """
            return self.action_code


    def normalize_errors(self, errs_goodsites , errs_badsites , all_errors = None , all_sites = None , TiersOnly=False):
        """
        converting the good/bad sites/errors matrix to numpy array. it creates an array with dim #errors X #sites X 2
        :param dict errs_goodsites: map of good sites and number of failed jobs in each site
        :param dict errs_goodsites: map of bad sites and number of failed jobs in each site
        :param list all_errors: the list of all errors. if is not given, the list from parent tasks object is used
        :param list all_sites: the list of all sites. if is not given, the list from parent tasks object is used
        :param bool TiersOnly: if it is true, site names are ignored and they are just categorized as T0, T1, T2, T3 and Others(just in case)
        """
        if not all_errors:
            all_errors = self.tasks.all_errors
        if not all_sites:
            all_sites = self.tasks.all_sites

        if TiersOnly :
            self.error_tensor = np.zeros( (len(all_errors) , 5 ,2 ) )
        else :
            self.error_tensor = np.zeros( (len(all_errors) , len(all_sites) ,2 ) )
        for good_bad,errs in {0:errs_goodsites,1:errs_badsites}.items() :       
            for err in errs :
                errcode = all_errors.index( int(err) )
                sites = self.error_tensor[errcode]
                for site in errs[err]:
                    if TiersOnly:
                        try:
                            site_index = int( site[1] )
                        except ValueError as err:
                            site_index = 4
                    else:
                        site_index = all_sites.index( site )
                    count = errs[err][site]
                    sites[site_index][good_bad] +=  count
    
    def __init__(self , tsk , name , tasks ):
        """
        initialize the Task object.
        :param dict tsk: the dictionary from json file, which includes "parameters" and "errors" keys
        :param str name: the name of the task
        :param Tasks tasks: the parent tasks object
        """
        self.Name = name
        self.tasks = tasks
        
        if "parameters" in tsk.keys():
            params = tsk["parameters"]
            self.action = self.Action( params , tasks.all_actions )
        else:
            self.action = None

        if "errors" in tsk.keys() :
            errors = tsk["errors"]
            goodsts = errors["good_sites"]
            badsts = errors["bad_sites"]
            self.normalize_errors( goodsts , badsts , TiersOnly=tasks.TiersOnly )


    def Get2DArrayOfErrors(self , force = True):
        """
        Converts the 3D numpy array to 2d array of errors by summing over the sites.
        :param bool force: force to calculate it, even it has been already calculated
        :ret numpy.array: a 2D numpy array where the first dimention indicates the index of the error and the second dimention has only size of two : bad_sites/good_sites
        """
        if not hasattr( self , "sum_over_sites" ) or force:
            self.sum_over_sites = np.sum( self.error_tensor , axis=1 )

        return self.sum_over_sites
            
    def GetInfo(self , labelsOnly = False):
        """
        Converts the task to a one dimention list. 
        :param bool labelsOnly: if it is true, only the header which includes the name of the fields is returned
        """
        if labelsOnly:
            return ['tsk' , 'action' , 'nErrors' , 'nErrorsInGoodSites' , 'nErrorsInBadSites' , 'DominantErrorCodeInGoodSites' , 'DominantErrorCodeInBadSites' , 'DominantBadSite' , 'DominantGoodSite']
        info = [ self.Name ,
                 self.action.code() ,
                 np.sum( self.error_tensor ) ]
    
        info.extend( np.sum( self.error_tensor , axis=(0,1) )  )

        if not hasattr( self , "sum_over_sites" ):
            self.sum_over_sites = np.sum( self.error_tensor , axis=1 )
        dominantErrors = np.argmax( self.sum_over_sites , axis=0 )
        info.extend([self.tasks.all_errors[i] for i in dominantErrors])

        sum_over_error = np.sum( self.error_tensor , axis=0 )
        dominantSite = np.argmax( sum_over_error , axis=0 )
        info.extend(dominantSite)
        
        return info


class Tasks :
    """
    a class to read 'actionhistory' json file and convert it to numpy array
    it involves two loops over the actions. in the first loop it extracts the name of all the sites and actions and also the error codes.
    in the second loop, for each entry a Task item is created and stored
    """
    def __init__(self, _file , binary=False , TiersOnly=False, all_sites=[] , all_errors=[] , all_actions=[]):
        """
        initialize an instance of Taks
        :param str _file: the full path of the actionhistory json file
        :param bool binary: if true, converts actions to acdc/non-acdc
        :param bool TiersOnly: if true, only the tier index of the site is stored instead of the full name
        :param all_actions, all_errors, all_actions: to be able to add additional values to the list
        """
        self.TiersOnly = TiersOnly
        self.IsBinary = binary
        self.fIn = open(_file)
        self.js = json.load( self.fIn )

        self.all_sites = all_sites
        self.all_errors = all_errors
        self.all_actions = all_actions

        
        self.FillSiteErrors()

        if binary :
            self.all_actions = ["non-acdc", "acdc"]
        
        self.AllData = []

        for tsk in self.js :
            self.AllData.append( Task( self.js[tsk] , tsk , self ) )

        self.ErrorsGoodBadSites = np.array( [ tsk.Get2DArrayOfErrors() for tsk in self.AllData ] )
        self.AllActions = np.array( [tsk.action.code() for tsk in self.AllData ] )
        self.df = DataFrame(data=[tsk.GetInfo() for tsk in self.AllData] , columns=self.AllData[0].GetInfo(True))

    def GetTrainTestDS(self , train_ratio ):
        """
        convert the information to train/test
        :param float train_ratio: number between 0 and 1, the fraction to go for the training
        :ret: train_x, train_y, test_x , test_y
        """
        n = int(train_ratio*len(self.AllData))
        return self.ErrorsGoodBadSites[:n] , self.AllActions[:n] , self.ErrorsGoodBadSites[n:] , self.AllActions[n:]
        
    def FillSiteErrors(self , Print=False):
        """
        For the first loop and fill the lists of errors, sites and actions
        :param bool Print: do printing after it has been done
        """
        for tsk in self.js :
            errors = self.js[tsk]["errors"]
            for site_status in ["good_sites" , "bad_sites" ] :
                sites = errors[site_status]
                for err in sites :
                    if int(err) not in self.all_errors:
                        self.all_errors.append(int(err))
                    for site in sites[err]:
                        if site not in self.all_sites :
                            self.all_sites.append( site )
            action = self.js[tsk]['parameters']['action']
            if action not in self.all_actions :
                self.all_actions.append( str(action) )
        self.all_sites.sort()
        self.all_errors.sort()
        self.all_actions.sort()

        if Print:
            print(self.all_sites)
            print(self.all_errors)
            print(self.all_actions)

    def PlotCorrelation(self):
        """
        produce and show the correlation plot, based on the output of GetInfo method of the Task object
        """
        plt.matshow(self.df.corr())
        plt.show()

    def GroupBy( self,  var1 , var2 ):
        """
        group by var1 and var2 and plot the counts
        """
        groupby = self.df.groupby([var1 , var2])
        var3 = "nErrorsInGoodSites" if "nErrorsInBadSites" in [var1,var2] else "nErrorsInBadSites"
        df_action_error_count = groupby[var3].count().reset_index()
        df_action_error_count.plot.scatter(x=var1 , y=var2 , s=df_action_error_count[var3] )
        plt.show()

