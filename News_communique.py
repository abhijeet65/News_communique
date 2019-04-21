import time
import sys

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

with open('news', 'r') as f:
    text = f.read()
    news = text.split("\n\n")
    v=0
    count = {'sport': 0, 'world': 0, "us": 0, "business": 0, "health": 0, "entertainment": 0, "sci_tech": 0}
    print("Loading approx ~32k News Articles from raw data So,wait for 5 minutes")
    for i in range(10):
            print("Loading" + "." * i)
            sys.stdout.write("\033[F")
            time.sleep(1)
    for news_item in news:
        lines = news_item.split("\n")
        #print(lines[6])
        file_to_write = open('data/' + lines[6] + '/' + str(count[lines[6]]) + '.txt', 'w+')
        count[lines[6]] = count[lines[6]] + 1
        file_to_write.write(news_item)  # python will convert \n to os.linesep
        file_to_write.close()
    print(count)

import pandas
import glob

category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
directory_list = ["data/sport/*.txt", "data/world/*.txt","data/us/*.txt","data/business/*.txt","data/health/*.txt","data/entertainment/*.txt","data/sci_tech/*.txt",]

text_files = list(map(lambda x: glob.glob(x), directory_list))
text_files = [item for sublist in text_files for item in sublist]

training_data = []


for t in text_files:
    f = open(t, 'r')
    f = f.read()
    t = f.split('\n')
    training_data.append({'data' : t[0] + ' ' + t[1], 'flag' : category_list.index(t[6])})

training_data[0]

training_data = pandas.DataFrame(training_data, columns=['data', 'flag'])
training_data.to_csv("train_data.csv", sep=',', encoding='utf-8')
print("Total data in training set ",training_data.data.shape)
l = ['sport', 'world', "us", "business", "health", "entertainment", "sci_tech"]

for i in range(0,7):
    place =  (training_data.flag == i)
    count[l[i]] = training_data[place].count()
print(count)
#place2 =(training_data.flag == 1)
#print(place2)

import pickle
from sklearn.feature_extraction.text import CountVectorizer


#GET VECTOR COUNT
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)

#SAVE WORD VECTOR
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))

from sklearn.feature_extraction.text import TfidfTransformer

#TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#SAVE TF-IDF
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))

# Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#clf = MultinomialNB().fit(X_train_tfidf, training_data.flag)
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf = MultinomialNB().fit(X_train, y_train)

#SAVE MODEL
pickle.dump(clf, open("nb_model.pkl", "wb"))

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

docs_new = "Messi joins other football team"
docs_new = [docs_new]

#LOAD MODEL
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("nb_model.pkl","rb"))

X_new_counts = loaded_vec.transform(docs_new)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)

print(category_list[predicted[0]])

import matplotlib.pyplot as plt
accuracy=[]
f1=[]
prec=[]
rec=[]

predicted = loaded_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test,predicted))
a=(accuracy_score(y_test,predicted))
print('F1_Score:',f1_score(y_test, predicted, average="macro"))
b=(f1_score(y_test, predicted, average="macro"))
print('Precision Score:',precision_score(y_test, predicted, average="macro"))
c=(precision_score(y_test, predicted, average="macro"))
print('Recall Score:',recall_score(y_test, predicted, average="macro"))
d=(recall_score(y_test, predicted, average="macro"))


accuracy.append(a)
f1.append(b)
prec.append(c)
rec.append(d)

result_bayes = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_bayes.to_csv('res_bayes.csv', sep = ',')

for predicted_item, result in zip(predicted, y_test):
    print(category_list[predicted_item], ' - ', category_list[result])

#graph implementation

left = [1, 2, 3, 4]


# heights of bars

height = [a, b, c, d]


# labels for bars

tick_label = ['Accuracy', 'F1_Score', 'Precision_Score', 'Recall_Score']


# plotting a bar chart

plt.bar(left, height, tick_label = tick_label,

        width = 0.8, color = ['red', 'green'])


# naming the x-axis

plt.xlabel('x - axis')
# naming the y-axis

plt.ylabel('y - axis')
# plot title

plt.title('My bar chart for Multinomial Naive Bayes')


# function to show the plot
plt.show()

from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_test,predicted)
print(confusion_mat)

from sklearn.neural_network import MLPClassifier

clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)

clf_neural.fit(X_train, y_train)

pickle.dump(clf_neural, open("softmax.pkl", "wb"))

predicted = clf_neural.predict(X_test)
print('Accuracy:', accuracy_score(y_test,predicted))
a=(accuracy_score(y_test,predicted))
print('F1_Score:',f1_score(y_test, predicted, average="macro"))
b=(f1_score(y_test, predicted, average="macro"))
print('Precision Score:',precision_score(y_test, predicted, average="macro"))
c=(precision_score(y_test, predicted, average="macro"))
print('Recall Score:',recall_score(y_test, predicted, average="macro"))
d=(recall_score(y_test, predicted, average="macro"))

accuracy.append(a)
f1.append(b)
prec.append(c)
rec.append(d)


result_softmax = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_softmax.to_csv('res_softmax.csv', sep = ',')

for predicted_item, result in zip(predicted, y_test):
    print(category_list[predicted_item], ' - ', category_list[result])

#graph implementation

left = [1, 2, 3, 4]


# heights of bars

height = [a, b, c, d]


# labels for bars

tick_label = ['Accuracy', 'F1_Score', 'Precision_Score', 'Recall_Score']


# plotting a bar chart

plt.bar(left, height, tick_label = tick_label,

        width = 0.8, color = ['red', 'green'])


# naming the x-axis

plt.xlabel('x - axis')
# naming the y-axis

plt.ylabel('y - axis')
# plot title

plt.title('My bar chart for MLP Classifier ')


# function to show the plot
plt.show()

from sklearn import svm
clf_svm = svm.LinearSVC()
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf_svm.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_svm, open("svm.pkl", "wb"))

predicted = clf_svm.predict(X_test)
print('Accuracy:', accuracy_score(y_test,predicted))
a=(accuracy_score(y_test,predicted))
print('F1_Score:',f1_score(y_test, predicted, average="macro"))
b=(f1_score(y_test, predicted, average="macro"))
print('Precision Score:',precision_score(y_test, predicted, average="macro"))
c=(precision_score(y_test, predicted, average="macro"))
print('Recall Score:',recall_score(y_test, predicted, average="macro"))
d=(recall_score(y_test, predicted, average="macro"))

accuracy.append(a)
f1.append(b)
prec.append(c)
rec.append(d)


result_svm = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_svm.to_csv('res_svm.csv', sep = ',')
for predicted_item, result in zip(predicted, y_test):
    print(category_list[predicted_item], ' - ', category_list[result])

#graph implementation

left = [1, 2, 3, 4]


# heights of bars

height = [a, b, c, d]


# labels for bars

tick_label = ['Accuracy', 'F1_Score', 'Precision_Score', 'Recall_Score']


# plotting a bar chart

plt.bar(left, height, tick_label = tick_label,

        width = 0.8, color = ['red', 'green'])


# naming the x-axis

plt.xlabel('x - axis')
# naming the y-axis

plt.ylabel('y - axis')
# plot title

plt.title('My bar chart for SVM Classifier ')


# function to show the plot
plt.show()

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('abhijeetsaraf65', 'Fvq4yMoFx64uvDa49gKi')
trace1 = {
  "x": ["MN Naive Bayes", "MLP Classifer", "SVM"],
  "y": accuracy,
  "name": "Accuracy",
  "type": "bar"
}
trace2 = {
  "x": ["MN Naive Bayes", "MLP Classifer", "SVM"],
  "y": prec,
  "name": "Precision",
  "type": "bar"
}
trace3 = {
  "x": ["MN Naive Bayes", "MLP Classifer", "SVM"],
  "y": rec,
  "name": "Recall",
  "type": "bar"
}
trace4 = {
  "x": ["MN Naive Bayes", "MLP Classifer", "SVM"],
  "y": f1,
  "name": "F1 Score",
  "type": "bar"
}
data = Data([trace1, trace2, trace3, trace4])
layout = {"barmode": "group"}
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
