<center><b><h1>Term Project</h1></bold></center>
<center><bold><h2>Mobile Phone Recommendation</h2></bold></center>

>Name: Naik Vishwa Chetankumar<br>
>UTA ID: 1001871311

Aim of this project:<br>
---

The Aim of a project to recommend a mobile phone using classification methods on 1.4 million cell phone reviews dataset.I have used three classifiers. 

<ul><li>Linear Support vector machine</li>
<li>Rigid classifier</li>
<li>Navie Bayes<br></li></ul>
<h2>Dataset Used</h2>
<ul><li>1.4 million cell phone reviews</li>
<li><a href="https://www.kaggle.com/masaladata/14-million-cell-phone-reviews">https://www.kaggle.com/masaladata/14-million-cell-phone-reviews</a></li></ul>

<h2>Libraries used in project</h2>
<ul>
<li>train_test_split:To split data in to train and test.(I have split data in to 80:20 ratio)</li>
<li>TfidfVectorizer:Convert a collection of raw documents to a matrix of TF-IDF features.</li>
<li>TfidfTransformer:Transform a count matrix to a normalized tf or tf-idf representation</li>
<li>MultinomialNB: The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may</li>
<li>RidgeClassifier:This classifier first converts the target values into {-1, 1} and then treats the problem as a regression task (multi-output regression in the multiclass case).</li>
<li>LinearSVC:Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.</li>
<li>accuracy_score:In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.</li>
<li>nltk:1.word_tokenize2.stop words</li>


Method:
---
Here , I have performed my project on google colaboratory but if you want to perform on jypeter make sure you have installed all the libraies which I have used.

1.Used pandas datafram to read dataset.(data.csv)

2.Perfomed data pre processing.<br>
Dara preprocessing is most crucial task in text analysis as well as in deep learning tasks. 
If you do not perform any pre processing on your raw data, then it will affect your model's performance and to your final result. Generally, in this step we remove emojis, stop words, some rare occurring words from text.
Also lower capitalization and tokenization are performed.
In preprocessing, I removed all the rows having no values in comment column, as without comment value there is nothing to predict.
To remove stop words from comments, first, I tokenized those comments using tokenizer tool provided by Natural Language toolkit (NLTK), and then removed those stop words. In nltk, there is one class "corpus" which contains list of these stop words. Along with them, comments are also converted into lower alphabets using lower() function.<br>

3.Data Visualization.<br>
For data Visualization I have used matplotlib, seaborn libraries. Data is postulated by heatmap, bar graph.

4.Divide data in to train and test. For this I have used Train-Test-Split function.

5.Model for classification.<br>
From many classification models like:Naive Bayes, Support vector machine, random forest, ridge regression, linear regression, etc.
I have used linear support vector machine, Navie bayes and rigid classifier.

6.Performance evaluation of algorithms:
We must know how our algorithm is working. For that purpose, some accuracy measures, error measurement techniques are used.
Such as, confusion matrix, F1 score, precision, recall, etc.

Important Links:
---
1.   GitHub: <a href="https://github.com/vishwanaik15/DataMining_Project">https://github.com/vishwanaik15/DataMining_Project</a><br>
2.   Youtube video:<br>
3.   My website: <a href="https://vishwanaik15.github.io/Acedemic/">https://vishwanaik15.github.io/Acedemic/</a><br>
4.   Blog: <a href="https://vishwanaik15.github.io/Acedemic/blog3.html"></a>
5.   Notebook: <a href="https://colab.research.google.com/drive/19B-tG-UV6mu9Im7TkJfcEXBcqBDw-aAt?usp=sharing">https://colab.research.google.com/drive/19B-tG-UV6mu9Im7TkJfcEXBcqBDw-aAt?usp=sharing</a>

<br>


<h2>Classifiers in this project:</h2>

>1.Linear Support Vector<br>
>2.Ridge Classifier<br>
>3.Naive Bayes<br>


Difference over References:
---
The reference I have used has not done much with dataset used torch for the accuracy and classification.<br>
<ul><li>Changes</li></ul>
<ul><li>Dataset has 6 csv files with data I have merged all the file to generate big dataset of size 921 megabutes.</li>
<li>I used TFIDFVectorizer.</li>
<li>I have used different classifer from skitlearn libraries for reccomendation analysis.</li>
<li>Reference:<a href="https://www.kaggle.com/satyamkryadav/multiclass-mobilesentiments-83">https://www.kaggle.com/satyamkryadav/multiclass-mobilesentiments-83</a></li>

Challenged Faced:
---
The dataset was having different csv files I merges all files in single and made a dataset big enoughr for analysis.<br>

The dataset was too large for the systems we are using, so I need to pre process data first in order to use it for training and testing. For pre processing I had to use case lowering, tokenization, regular expression for alphabets, punctuation removal and removing stop words.<br>

The reference I used has torch but I used Clssifieres.






Contributions
---



1. I hae used three classifier and then used the best one with highest accuracy for the final reccomendation system.<br>
2. For my large dataset I have found linear support vector machine with highest accuracy so I uset it for final reccomendation.<br>
3. Saved model in local system for further any classification purpose.<br>
4. Made a application prototype for future use.<a href="https://proto.io/">https://proto.io/</a><br>

<br>


References:
---
1.https://www.kaggle.com/satyamkryadav/multiclass-mobilesentiments-83<br>
2.Dataset Reading:https://towardsdatascience.com/loading-large-datasets-in-pandas-11bdddd36f7b<br>
3.Data preprocessing:https://scikit-learn.org/stable/modules/preprocessing.html<br>
4.Text classification:https://www.nltk.org/<br>
5.Vectorizing using Scikit-learn API's:https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html<br>
6.Vectorizing using Scikit-learn API's:https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html<br>
7.Naive Bayes:https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html<br>
8.Linear support vector Machine:https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html<br>
9.Rigid Classifier:https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html<br>
10.Confusion Matrix:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html<br>
11.Accuracy score:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html<br>
12.F1 Score:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html<br>
13.Heatmap: https://seaborn.pydata.org/<br>
14.Application Sketch:https://proto.io/<br>









