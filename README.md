# AIErrorHandling
Tools to produce suggestion for the CMS workflow failures recovery panel. 
The idea is to train neural networks based on the previous actions and the details of the failed jobs. 

There are two subdirectories :
## training
all the codes that are used for training should go here. For the moment there are two approaches for the training : 
### Based on the error codes and site names and statuses
codes are added under the training/SitesErrorCodes/ codes
### Add the log/error files information for training
a new sub-package under the training can be created under the training directory
## models
API to retrieve the suggestion values
