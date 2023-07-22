### Importing Libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy
from sklearn.metrics import classification_report,accuracy_score

### Importing Libraries for Outlier Detection"""

from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM

"""### Reading our Dataset"""

from google.colab import drive
drive.mount('/content/drive')

df= pd.read_csv("/content/drive/MyDrive/creditcard.csv")

df.head()

"""### Data Analysis"""

df.shape

"""#### Checking Null Values"""

df.isnull().sum()

"""### Checking the distribution of Normal and Fraud cases in our Data Set"""

fraud_check = pd.value_counts(df['Class'], sort = True)
fraud_check.plot(kind = 'bar', rot=0, color= 'r')
plt.title("Normal and Fraud Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
 ## Defining labels to replace our 0 and 1 valuelabels= ['Normal','Fraud']
## mapping those labels
plt.xticks(range(2), labels)
plt.show()

"""
#### Let us see what is the shape of Normal and Fraud data set"""

fraud_people = df[df['Class']==1]
normal_people = df[df['Class']==0]

fraud_people.shape

normal_people.shape

"""#### Finding out the avg amount in our both the data sets"""

fraud_people['Amount'].describe()

normal_people['Amount'].describe()

"""#### Let us analyse it visually"""

graph, (plot1, plot2) = plt.subplots(2,1,sharex= True)
graph.suptitle('Average amount per class')
bins = 70

plot1.hist(fraud_people['Amount'] , bins = bins)
plot1.set_title('Fraud Amount')

plot2.hist(normal_people['Amount'] , bins = bins)
plot2.set_title('Normal Amount')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show();

"""#### Plotting a corr Heatmap"""

df.corr()
plt.figure(figsize=(30,30))
g=sns.heatmap(df.corr(),annot=True)

"""### Creating our Dependent and Independent Features"""

columns = df.columns.tolist()
# Making our Independent Features
columns = [var for var in columns if var not in ["Class"]]
# Making our Dependent Variable
target = "Class"
x= df[columns]
y= df[target]

x.shape

y.shape

x.head() ## Independent Variable

y.head() ## Dependent Variable

"""## Model building

### Splitting the data
"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

"""### We wil be using the following Models for our Anamoly Detection:
- Isolation Forest
- OneClassSVM

## Isolation Forest

#### One of the newest techniques to detect anomalies is called Isolation Forests. The algorithm is based on the fact that anomalies are data points that are few and different. As a result of these properties, anomalies are susceptible to a mechanism called isolation.

#### This method is highly useful and is fundamentally different from all existing methods. It introduces the use of isolation as a more effective and efficient means to detect anomalies than the commonly used basic distance and density measures. Moreover, this method is an algorithm with a low linear time complexity and a small memory requirement. It builds a good performing model with a small number of trees using small sub-samples of fixed size, regardless of the size of a data set.

#### Typical machine learning methods tend to work better when the patterns they try to learn are balanced, meaning the same amount of good and bad behaviors are present in the dataset.

#### How Isolation Forests Work The Isolation Forest algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The logic argument goes: isolating anomaly observations is easier because only a few conditions are needed to separate those cases from the normal observations. On the other hand, isolating normal observations require more conditions. Therefore, an anomaly score can be calculated as the number of conditions required to separate a given observation.

#### The way that the algorithm constructs the separation is by first creating isolation trees, or random decision trees. Then, the score is calculated as the path length to isolate the observation
"""

iso_forest= IsolationForest(n_estimators=100, max_samples=len(x_train),random_state=0, verbose=0)

iso_forest.fit(x_train,y_train)

ypred= iso_forest.predict(x_test)

ypred

"""#### Mapping the values as we want to have an output in 0 and 1"""

ypred[ypred == 1] = 0
ypred[ypred == -1] = 1

"""### Accuracy score and Matrix"""

print(accuracy_score(y_test,ypred))

print(classification_report(y_test,ypred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, ypred)

"""### We can also print how many errors our model have"""

n_errors = (ypred != y_test).sum()
print("Isolation Forest have {} errors.".format(n_errors))

"""## OneClassSVM"""

svm= OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, max_iter=-1)

svm.fit(x_train,y_train)

ypred1= svm.predict(x_test)

"""#### Here also we do the same thing as above, mapping our results in 0 and 1"""

ypred1[ypred1 == 1] = 0
ypred1[ypred1 == -1] = 1

"""### Accuracy score and Matrix"""

print(accuracy_score(y_test,ypred))

print(classification_report(y_test,ypred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, ypred)

n_errors = (ypred1 != y_test).sum()
print("SVM have {} errors.".format(n_errors))

"""## Solving the Problem Statement using PyCaret Library(Auto ML)

# PyCaret :

### PyCaret is an open source, low-code machine learning library in Python that allows you to go from preparing your data to deploying your model within minutes in your choice of notebook environment.

### Installing Pycaret
"""

pip install pycaret

df= pd.read_csv("creditcard.csv")

df.head()

from pycaret.classification import *

model= setup(data= df, target= 'Class')

compare_models()

random_forest= create_model('rf')

"""### As we see we have a very good Kappa score which is often seen in an Imbalanced dataset"""

random_forest

"""### We can Hypertune our model to"""

tuned_model= tune_model('random_forest')

"""## Predictions"""

pred_holdout = predict_model(random_forest,data= x_test)

pred_holdout

