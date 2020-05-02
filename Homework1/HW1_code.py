# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:14:33 2018

@author: Kuo
"""
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == '__main__':
    rawData = pd.read_csv('character-deaths.csv') #load data
    rawData = rawData.fillna(0)  #Nan value been replaced by zero
    
    #preserve only one column from (Death Year , Book of Death , Death Chapter)
    rawData.drop(['Book of Death', 'Death Chapter', 'Name'], axis=1, inplace=True)

    #convert the value of column 'Death year' that greater than zero into one
    rawData.loc[rawData['Death Year'] > 0, 'Death Year'] = 1

#    convert column 'Allegiances' into one hot (1)
#    rawData['Allegiances'] = rawData['Allegiances'].str.get_dummies().values.tolist()
#    convert column 'Allegiances' into one hot (2)
    rawData = pd.concat([rawData, pd.get_dummies(rawData['Allegiances'], prefix='Allegiances')],axis=1)
    rawData.drop(['Allegiances'],axis=1, inplace=True)
    
    #split data into training dataset, testing dataset (1)
    train = rawData.sample(frac=0.75, random_state=25)  
    test = rawData.drop(train.index)
    #split data into training dataset, testing dataset (2)
#    from sklearn.model_selection import train_test_split
#    train, test = train_test_split(rawData, test_size=0.2)
    
    train_x = train.drop(['Death Year'], axis=1)
    train_y = train['Death Year']
    
    test_x = test.drop(['Death Year'], axis=1)
    test_y = test['Death Year']
        
    clf = tree.DecisionTreeClassifier(random_state=385, max_depth=15)
    clf = clf.fit(train_x, train_y)
#    print('Training Accuracy：',clf.score(train_x, train_y))
    
    import graphviz 
    dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=train_x.columns,  
                         class_names=train_y.name,  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render()
    
    y_pred = clf.predict(test_x)
    test_y = list(test_y)
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
#    print(confusion_matrix(test_y, y_pred))
#    print('tn, fp, fn, tp = ', confusion_matrix(test_y, y_pred).ravel())
    
    cnf_matrix = confusion_matrix(test_y, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["alive","dead"])
    plt.show()
    print('****** training data ******')
    print('accuracy = ', clf.score(train_x, train_y), '\n')
    print('****** testing data ******')
    print('precision = ', precision_score(test_y, y_pred))
    print('recall = ', recall_score(test_y, y_pred))
    print('accuracy = ', accuracy_score(test_y, y_pred))
    