# Credit_Risk_Analysis
I was tasked to use Python to build and evaluate several machine learning models to predict credit risk. This machine learning model is a supervised one, meaning that it use data with labels. Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the ClusterCentroids algorithm. Then, I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 

## Random Oversampling
In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. 
