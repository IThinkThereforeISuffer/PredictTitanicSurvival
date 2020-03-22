import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

def updateSex(people):
  for i in range(len(people)):
    if (people[i] == "male"):
      people[i] = 0
    else:
      people[i] = 1
  return people

def calculateMeanAge(people):
  nominator = 0.0
  denominator = 0
  for i in range(len(people)):
    if (math.isnan(float(people[i]))):
      pass
    else:
      nominator += float(people[i])
      denominator += 1   
  mean = nominator/denominator
  return "{0:.1f}".format(mean)

def updateAge(people):
  meanAge = calculateMeanAge(people)
  for i in range(len(people)):
    if (math.isnan(float(people[i]))):
      people[i] = meanAge
  return people

def updateClass(people, klass):
  for i in range(len(people)):
    if (people[i] == klass):
      people[i] = 1
    else:
      people[i] = 0
  return people
# Load the passenger data
passengers = pd.read_csv('passengers.csv')

# Update sex column to numerical
passengers['Sex'] = updateSex(passengers['Sex'])

# Fill the nan values in the age column
passengers['Age'] = updateAge(passengers['Age'])

# Create a first class column
passengers['FirstClass'] = passengers["Pclass"]
passengers['FirstClass'] = updateClass(passengers['FirstClass'], 1)

# Create a second class column
passengers['SecondClass'] = passengers["Pclass"]
passengers['SecondClass'] = updateClass(passengers['SecondClass'], 2)

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
train_features, test_features, train_survival, test_survival = train_test_split(features, survival, test_size = 0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Create and train the model
model = LogisticRegression()
model.fit(train_features, train_survival)

# Score the model on the train data
train_score = model.score(train_features, train_survival)
# Score the model on the test data
test_score = model.score(test_features, test_survival) 
# Analyze the coefficients
name_and_coeffs = list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0]))

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([1.0,22.0,1.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)

# Make survival predictions!
boolean_survival = model.predict(sample_passengers)
probability_survival = model.predict_proba(sample_passengers)
print(boolean_survival)
print(probability_survival)
