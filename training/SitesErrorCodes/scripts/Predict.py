from AIErrorHandling.models.SiteErrorCodeModelLoader import *
from AIErrorHandling.models import Predictor


import json
jj = json.load( open("../data/actionshistory.json") )
for a in jj :
    jjb = jj[a]['errors']
    #print(jjb)
    print( Predictor( **jjb  ) )
