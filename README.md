# Classification of News HeadLines

News Headline Classification through multiple machine learning model and comparison of results.

Models implemented:

 * Multinomial Naive Bayes 
 * Support Vector Machines 
 * Neural Network with Softmax Layer
 

Metrics used to evaluate the performance of models:

 * Accuracy
 * Precision
 * Recall
 * F1 Score 
 
 We evaluate each classifier's ability to select the appropriate category given an article’s title and a brief article description. The confusion matrix is created to  explore the results and calculate the metrics. 



###### Feature Extraction Techniques:
The collection of text documents is converted to a matrix of token counts using count vectorize that produces a sparse representation of the counts.

TFIDF,term frequency–inverse document frequency, is the statistic that is intended to reflect how important a word is to a document in our corpus. This is used to extract the most meaningful words in the Corpus.
corpus ( Collection of written texts,words etc. )


Our Datasets is a collection of datasets of short text fragments that we used for the evaluation of  our topic-based text classifier. This is a dataset of  ~32K english news extracted from RSS feeds of popular newspaper websites (nyt.com, usatoday.com, reuters.com). Categories are: Sport, Business, U.S., Health, Sci&Tech, World and Entertainment.

DataSet is Included in the project Folder as news (raw data)


Packages required: 

 * Pandas
 * sklearn
 * Numpy
 
# Steps to Run :   

 
1.Open Cmd   
2.Open Jupyter Notebook by Typing Jupyter Notebook in cmd. (See Installation details for Jupyter if not already Installed!!)   
3.open News Classification.ipynb   
4.Run the program blocks one at a time for clear visuallization and categorization of our implementation for classifying news headlines  
link for docs ![https://docs.google.com/document/d/1VptDVOIk1nrRSKbRwtTzWb26rauhiZwLNMaqw4zOfcw/edit?usp=sharing]  

#####Contributions:

###Algorithm Implementation

* Neural Network-Abhijeet Saraf
* Multinomial Naive Bayes-Vishrut Vats
* Support Vector Machines-Abhishek Kumar Mandal

###Documentation:

* Abhijeet Saraf and Vishrut Vats

###Debugging:

* Abhishek Kumar Mandal and Vishrut Vats

#Visualization:

* Abhijeet Saraf







