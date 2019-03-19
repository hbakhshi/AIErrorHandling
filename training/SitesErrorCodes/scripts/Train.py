from AIErrorHandling.training.SitesErrorCodes import *
#from AIErrHand.Tasks import *
#from AIErrHand.DNNTrain import *
import numpy
import random
import tensorflow as tf
dir()
random_seed = random_seed
random.seed(random_seed)
numpy.random.seed(random_seed)
tf.set_random_seed(random_seed)

#tasks = Tasks( "../data/history.180618.json" )
tasks = Tasks.Tasks( "../data/actionshistory.json" , True )
trainer = DNNTrain.DNNTrain( tasks , 0.85)
trainer.MakeModel(loss='binary_crossentropy' , layers=[(100,None,'tanh'),(100,None,'relu'),(200,None,'relu')])
fit_res = trainer.Fit(batch_size=2000 , epochs=20 , validation_split=0.25)
print(fit_res)
res = trainer.Test()
trainer.SaveModel( "FirstTry" , 1 , 1 )

