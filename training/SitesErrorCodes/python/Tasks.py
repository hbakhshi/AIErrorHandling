import workflowwebtools as wwt
from pandas import DataFrame

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras


class Task:
    class Action:
        def __init__(self,parameters, all_actions):
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
            try :
                self.action_code = all_actions.index( self.action )
            except ValueError as err :
                if all_actions[ non_acdc_index ] == "non-acdc" :
                    self.action_code = non_acdc_index
                else :
                    raise err
            
        def code(self):
            return self.action_code


    def normalize_errors(self, errs_goodsites , errs_badsites , all_errors = None , all_sites = None , TiersOnly=False):
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
        self.Name = name
        params = tsk["parameters"]
        self.action = self.Action( params , tasks.all_actions )
        errors = tsk["errors"]
        goodsts = errors["good_sites"]
        badsts = errors["bad_sites"]

        self.tasks = tasks
        
        self.normalize_errors( goodsts , badsts , TiersOnly=tasks.TiersOnly )

    def Get2DArrayOfErrors(self , force = True):
        if not hasattr( self , "sum_over_sites" ) or force:
            self.sum_over_sites = np.sum( self.error_tensor , axis=1 )

        return self.sum_over_sites
            
    def GetInfo(self , labelsOnly = False):
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
    def __init__(self, _file , binary=False , TiersOnly=False, all_sites=[] , all_errors=[] , all_actions=[]):
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
        n = int(train_ratio*len(self.AllData))
        return self.ErrorsGoodBadSites[:n] , self.AllActions[:n] , self.ErrorsGoodBadSites[n:] , self.AllActions[n:]
        
    def FillSiteErrors(self):       
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

        print(self.all_sites)
        print(self.all_errors)
        print(self.all_actions)

    def PlotCorrelation(self):
        plt.matshow(self.df.corr())
        plt.show()

    def GroupBy( self,  var1 , var2 ):
        groupby = self.df.groupby([var1 , var2])
        var3 = "nErrorsInGoodSites" if "nErrorsInBadSites" in [var1,var2] else "nErrorsInBadSites"
        df_action_error_count = groupby[var3].count().reset_index()
        df_action_error_count.plot.scatter(x=var1 , y=var2 , s=df_action_error_count[var3] )
        plt.show()

