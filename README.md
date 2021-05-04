<center><b><h1>Term Project</h1></bold></center>
<center><bold><h2>Mobile Phone Recommendation</h2></bold></center>

>Name: Naik Vishwa Chetankumar<br>
>UTA ID: 1001871311

Aim of this project:<br>
---

The Aim of a project to recommend a mobile phone using classifaction methods on 1.4 million cell phone reviews dataset.I have used three classifiers. 
<ul><li>Linear Support vector machine</li>
<li>Rigid classifier</li>
<li>>K nearest neighbours<br></li></ul>

<h2>Libraries used in project</h2>
<ul>
<li>train_test_split:</li>


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
I have used linear support vector machine, KNN and rigid classifier.

6.Performance evaluation of algorithms:
We must know how our algorithm is working. For that purpose, some accuracy measures, error measurement techniques are used.
Such as, confusion matrix, F1 score, precision, recall, etc.

Important Links:
---
1.   GitHub: 
2.   Youtube video:<br>
3.   My website: 
4.   Blog: 

<br>




<h2>Classifiers in this project:</h2>

>1.Linear Support Vector<br>
>2.Ridge Classifier<br>
>3.k nearest neighbours<br>


Difference over References:
---


Challenged Faced
---

The dataset was too large for the systems we are using, so I need to pre process data first in order to use it for training and testing. For pre processing I had to use case lowering, tokenization, regular expression for alphabets, punctuation removal and removing stop words.<br>

Dataset contains so many (above 70%) missing values (NaN) in comment column, which is required to predict rating. Before proceeding further removed these rows without affecting accuracy and meaningful data.<br>

I used Flask to write web application code for text classification. Flask is quite hard to understand, but what is more peculier than code is it's deployment on live server. Heroku is tool that I am using for deployment. There are so many required dependencies to make flask program compatible. It took me straight 3 hours for deployment.<br>

By using pipeline I could have got accuracy in less time, but it would not give precise information about hyperparameters, so I choose the other way. But achieved this results after so many crash of kernel.

<br>
Importing required dependencies and mounting google drive for data fetching
---



Contributions
---



1.   The reference that  i have used for support vector machine, it was done without using any hyperparameters. I have implemented it's probability and "linear" kernel and got improved accuracy. 
2.   Implemented Ensemble methods over some of selected classifiers. I achieved a good accuracy than those individuals are getting. I used voting classifier in ensemble methods, with appropriate weighted voting to three classifiers. As I have used less data for those classifiers, accuracy is less. But for larger data surely would get better results. 
3. Saved model in local system for further any classification purpose. 


<br>

Challenges faced:
---



1.   The dataset was too large for the systems we are using, so I need to pre process data first in order to use it for training and testing. For pre processing I have to use case lowering, tokenization, regular expression for alphabets, punctuation removal and removing stop words.
2. Dataset contains so many (above 70%) missing values (NaN) in comment column, which is required to predict rating. Before proceeding further removed these rows without affecting accuracy and meaningful data.
3. I used Flask to write web application code for text classification. Flask is quite hard to understand, but what is more peculier than code is it's deployment on live server. Heroku is tool that I am using for deployment. There are so many required dependencies to make flask program compatible. It took me straight 3 hours for deployment. 
3. By using pipeline I could have done the same thing (getting accuracy) in less time, but it would not give precise information about hyperparameters, so I choose the other way. But achieved this results after so many crash of kernel. I am using 5-7 classifiers for accuracy comparisons and data analysis. So I have to get in-depth knowledge about those classifiers before implementation and for better data visualization, I need to understand 7-10 types of libraries and graphs.

<br>


References:
---


> I used sklearn library for classifiers as well as for vectorization and transformation, so most of the references are linked to official website of sklearn. 

1.   Reading large dataset: https://towardsdatascience.com/3-simple-ways-to-handle-large-data-with-pandas-d9164a3c02c1
2.   Data Preparation: https://www.kaggle.com/ngrq94/boardgamegeek-reviews-data-preparation
3.   Data preprocessing: https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/<br>
https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f 
4. Multinomial Naive Bayes: https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
5. Working with text data: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
6. Multi-Class Text Classification with Scikit-Learn: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
7. Vectorizing using Scikit-learn API's : https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
8. Ridge Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html
9. Linear SVC classifier: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
10. SVC classifier: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
11. Logistic regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
12. Ensemble methods: https://scikit-learn.org/stable/modules/ensemble.html
13. Confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
14. Classification report: https://scikit-learn.org/0.18/modules/generated/sklearn.metrics.classification_report.html
15. Random histograms: https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
16. Heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html







