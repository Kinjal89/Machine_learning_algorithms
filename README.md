# Machine_learning_algorithms
Language.csv: import pandas as pd
from sklearn.tree import DecisionTreeClassifier
language= pd.read_csv('Language.csv')
language
A = language.drop(columns=['morphology'])
b = language['morphology']
model = DecisionTreeClassifier()
model.fit(A, b)
prediction = model.predict([ [1200, 1], [250, 0] ])
prediction

Output: array(['RM', 'PM'], dtype=object)
******************************************************************************************************************************

To predict if a language is one with Rich morphology or poor morphology with ACCURACY_SCORE:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

language= pd.read_csv('Language.csv')
language
A = language.drop(columns=['morphology'])
b = language['morphology']
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=float(0.2))

model = DecisionTreeClassifier()
model.fit(A_train, b_train)
prediction = model.predict(A_test)
score = accuracy_score(b_test, prediction)
score

Output: 1.0
******************************************************************************************************************************


To predict if a language is one with Rich morphology or poor morphology to visualize DECISION TREES:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

case= pd.read_csv('case.csv')
K = case.drop(columns=['languages'])
m = case['languages']

model = DecisionTreeClassifier()
model.fit(K, m) #[We are only passing the training dataset]

tree.export_graphviz(model,out_file='music-recommender.dot', 
                     feature_names= ['geography', 'numberofcases'], 
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)


******************************************************************************************************************************


music.csv: To predict what a 21-year male and 22-year female prefer listening:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
music_data= pd.read_csv('music.csv')
music_data
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict([ [21, 1], [22, 0] ])
predictions

Output: array(['HipHop', 'Dance'], dtype=object)

******************************************************************************************************************************


case.csv: To predict the language and the number of case markings it could have based on Geography:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
case= pd.read_csv('case.csv')
case
K = case.drop(columns=['languages'])
m = case['languages']

model = DecisionTreeClassifier()
model.fit(K, m)
prediction = model.predict([ [11, 5], [10, 6] ])
prediction

Output: array(['Bengali', 'Russian'], dtype=object)

******************************************************************************************************************************


case.csv: To predict the language and the number of case markings it could have based on Geography with an ACCURACY_SCORE:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split # [This import function is to split training vs test data to calculate accuracy]
from sklearn.metrics import accuracy_score

case= pd.read_csv('case.csv')
K = case.drop(columns=['languages'])
m = case['languages']
K_train, K_test, m_train, m_test = train_test_split(K, m, test_size=0.2) 
#[This means that we have allow 20% of data to be test data, The train and test are input and output sets for training and testing]

model = DecisionTreeClassifier()
model.fit(K_train, m_train) #[We are only passing the training dataset]
predictions = model.predict(K_test) #[This is to test on the evaluation dataset]
score = accuracy_score(m_test, predictions)
score
Output: 1.0

******************************************************************************************************************************
