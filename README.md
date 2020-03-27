# SpamClassifier

A basic spam classifier which tests various algorithms for accuracy by training over 15000 samples

## About

- The classifier uses the enron email corpus dataset for email spam classification
- You can find the dataset at https://www.cs.cmu.edu/~./enron/
- The program uses [scikit learn] (https://scikit-learn.org/) python library to use the various machine learning algorithms to train the data 
- The users can use the detector test script to classify a mail into a spam or a ham mail.

## Tech/Framework used

This classifier uses:-

- [Multinomial Naive Bayes] (https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

- [Complement Naive Bayes] (https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)

- [Gaussian Naive Bayes] (https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

- [Bernoulli Naive Bayes] (https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)

- [LinearSVC] (https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

- [SVC] (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
