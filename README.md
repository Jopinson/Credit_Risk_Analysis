# Credit_Risk_Analysis
I was tasked to use Python to build and evaluate several machine learning models to predict credit risk. This machine learning model is a supervised one, meaning that it use data with labels. Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the ClusterCentroids algorithm. Then, I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.

## Random Oversampling
Class imbalance refers to a situation in which the existing classes in a dataset aren't equally represented. One method we used to get past this was to use random oversampling. In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. 
![oversampling](https://user-images.githubusercontent.com/68392225/99113309-a48d5980-25b4-11eb-885d-d13e7c490b7e.png)

Once we resampled the dataset 
* We ran an accuarcy test and got 63%. This is low, and we probably shouldn't trust the results from oversampling. 
* Our precision and our recall scores for people that are high risk for credit fraud are 0.01 and 0.59 
* Our precision and recall scores were imbalanced so we also have a low F1 score 0.02 
* Low risk category is closer with 1.00 and 0.68 and has a better F1 score of 0.81, but its still not a reliable test. 

## SMOTE Oversampling
The second method used to deal with imbalanced datasets was SMOTE. With SMOTE instead of randomly selecting data instances to balance them out, new instances are interpolated from a number of its closest neighbors. Based on the values of these neighbors, new values are created.
![SMOTE](https://user-images.githubusercontent.com/68392225/99116453-aa396e00-25b9-11eb-8fa3-2cdf87ca6e3b.png)

The results of our SMOTE oversampling were slightly better.
* Our balanced accuracy score was 65%, still not the best results.
* Our precision and our recall scores for people that are high risk for credit fraud are 0.01 and 0.67 
* Our precision and our recall scores for people that are at a low risk for credit fraud are 1.00 and 0.63
* Our high risk F1 score remained the same, and our low risk F1 score went down to 0.77

## Undersampling
Undersampling is another technique to address class imbalance. Instead of increasing the size of the minority sample, undersampling decreases the size of the majority. 
![undersampling](https://user-images.githubusercontent.com/68392225/99117923-33ea3b00-25bc-11eb-9260-eb3064e898d5.png)

The undersampling test was not as successful as the previous two tests.
* Our balanced accuracy score was 51%
* Our precision and our recall scores for people that are high risk for credit fraud are 0.01 and 0.63 
* Our precision and our recall scores for people that are at a low risk for credit fraud are 1.00 and 0.39
* Our high risk F1 score dropped to 0.01, and our low risk F1 score went down to 0.56

## SMOTEENN and Combination Sampling
SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN oversamples the minority class and clean the resulting data with an undersampling strategy.
![Combo](https://user-images.githubusercontent.com/68392225/99118429-1f5a7280-25bd-11eb-8719-af3e938739cd.png)

* Our balanced accuracy score was 64%
* Our precision and our recall scores for people that are high risk for credit fraud are 0.01 and 0.70 
* Our precision and our recall scores for people that are at a low risk for credit fraud are 1.00 and 0.57
* Our high risk F1 score rose to 0.02, and our low risk F1 score went up to 0.73
* Results improved from undersampling, but still not the highest or best results. 
