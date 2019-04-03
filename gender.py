from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# CHALLENGE - create 3 more classifiers...
# 1 decision tree
# 2 naive bayes
# 3 Support vector classification

print("Classifiers used: Decision tree, Naive Bayes, SVM")
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# decision tree
#variable to store the decision tree model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction = clf.predict([[181, 80, 44]])
print("decision tree prediction: ",prediction)

#naive bayes
gnb= GaussianNB()
gnb= gnb.fit(X,Y)
NB_pred= gnb.predict([[181,80,44]])
print("Naive bayes prediction: ", NB_pred)

#support vector classification
s= SVC()
s= s.fit(X,Y)
s_pred= s.predict([[181,80,44]])
print("SVM prediction: ",s_pred)

