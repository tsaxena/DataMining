
import csv
import numpy  as np
import pandas as pd
import math
from scipy.stats import mode
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from pandas import *


class TitanicClassifier(object):
  
    def __init__(self):
	print("Initializing the Classifier")
    	#read the titanic train data    
        trainpath 	= './Data/train.csv'
        testpath  	= './Data/test.csv'
        rawtraindf   	= pd.read_csv(trainpath)
        rawtestdf    	= pd.read_csv(testpath)
	self.traindf   	= self.cleandf(rawtraindf)
        self.testdf    	= self.cleandf(rawtestdf)  
        self.prio       = {}
        self.means 	= {}
	self.ssd 	= {}
	self.totals 	= 0
        self.klasses = [0, 1]
        self.categorical_feature_list = []
        self.numerical_feature_list  = []
	self.gender_prob = {}
        self.pclass_prob = {}
        self.embark_prob = {}

    #processing each column
    def cleandf(self, df): 
        #print("Clean the data")


	#print("Cleaning the fare data")
	# replace the 0 fares with the mean fare of the class
	# replace 0 with non a number
        # find the mean for the class 
	# replace all nan with the mean class fare
	df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
	classmeans = df.pivot_table(values='Fare', rows='Pclass', aggfunc=np.mean)
        # incredibly useful function
        # get the Fare and Pclass projection
        # applies row-wise to the projection
	df.Fare = df[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if np.isnan(x['Fare']) else x['Fare'], axis=1 )

        #print("Cleaning the age data")
	df.Age = df.Age.map(lambda x: np.nan if x==0 else x)
	meanAge  = np.mean(df.Age)
    	df.Age   = df.Age.map(lambda x: meanAge if np.isnan(x) else x)

        #print("Cleaning the embarkment data")
    	modeEmbarked = mode(df.Embarked)[0][0]
        df.embarked = df.Embarked.fillna(modeEmbarked)

	#print("Cleaning the Cabin data")
	df.Cabin = df.Cabin.fillna('Unknown')

	#print("Clean family data")
	#print(df['Ticket'])

        return df
    
	

    def train(self):
	print("Training the Classifier")
	#train features
	trainfeatures = self.traindf[['Pclass','Sex', 'Age','SibSp','Parch','Fare','Cabin','Embarked' ]] 
	trainlabels   = self.traindf[['Survived']]

        # calculate totals for both class
        self.totals = self.traindf.groupby('Survived').size()
	#print(self.totals)
	
	#for numeric features
        grouped = self.traindf[[ 'Survived','Age','SibSp','Parch','Fare']]
	self.numeric_feature_list = grouped.columns.values.tolist()
        self.means = grouped.groupby('Survived').mean()
        self.ssd   = grouped.groupby('Survived').std()
        #print(self.means.iloc[0]['Age'])	

	#for categorical features
        categorical_features = self.traindf[['Survived','Pclass','Sex', 'Embarked' ]] 
        self.categorical_feature_list = categorical_features.columns.values.tolist()
        
        self.gender_prob = categorical_features.groupby(['Survived', 'Sex']).size()
        #print(self.gender_prob)
        self.pclass_prob = categorical_features.groupby(['Survived', 'Pclass']).size()
        #print(self.pclass_prob)
        self.embark_prob = categorical_features.groupby(['Survived', 'Embarked']).size()
        #print(group_by_embark)
        
    
    def pdf(self, mean, ssd, x):
       sqrt2pi = math.sqrt(2 * math.pi)
       #Probability Density Function computing P(x|y)
       #input is the mean, sample standard deviation for all the items in y, and x.
       ePart = math.pow(math.e, -(x - mean)**2/(2*ssd**2))
       prob  = ((1.0 / (sqrt2pi*ssd)) * ePart)
       return math.log(prob)
    

    def calculate(self, x):
       
	#calculate the probabilies of the two
        logprob = Series(0, index=[0, 1]).astype('float')
         
        #for numeric features
        #'Age','SibSp','Parch','Fare'
        #print(self.numeric_feature_list)
        test_features =  self.numeric_feature_list[1::]
	#print(test_features)
        
        for k in self.klasses:
            #logprob.setdefault(k, 0)
	    #print('class name', k)
            for nf in test_features:
                logprob[k] += self.pdf(self.means.ix[k, nf], self.ssd.ix[k, nf], x.ix[nf])

        #print(logprob)
       
        #categorical data 
        #categorialFeatures = list(my_dataframe.columns.values)
        for k in self.klasses:
	    #print('class name', k)
            #'Pclass','Sex', 'Embarked'
            logprob[k] += math.log((self.gender_prob.ix[k, x.Sex] * 0.1) / self.totals[k])
            logprob[k] += math.log((self.pclass_prob.ix[k, x.Pclass] * 0.1) / self.totals[k])
	    logprob[k] += math.log((self.embark_prob.ix[k, x.Embarked] * 0.1) / self.totals[k])
       
        #print(logprob)
        #print(logprob.idxmax()) #of the probabilities of the classes
	return (logprob.idxmax())

    def predict(self):
        #prepare the file 
        print("Predicting test data")
	predictiondf = pd.DataFrame(self.testdf['PassengerId'])
        predictiondf['Survived']=[0 for x in range(len(self.testdf))]
        predictiondf.to_csv('./Data/first_submission.csv', index=False)

	#find the probability for each class
        predictiondf['Survived'] = self.testdf.apply(lambda row: self.calculate(row), axis=1)
 
        #write it to the output file
        prediction_path = './Data/first_submission.csv'
        prediction_csv  = pd.read_csv(prediction_path)
        prediction_csv['Survived']=predictiondf['Survived']
        prediction_csv.to_csv('./Data/first_submission.csv', index =False)



def main():
    print("Calling Naive Bayes")
    c = TitanicClassifier()
    c.train()
    c.predict()
    


if  __name__ =='__main__':
    main()



