import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle as c


def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print("saved")


def make_dict():
    direc = "E:/Development work/SpamClassifier/enron1/emails/"
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    words = []
    c = len(emails)
    for email in emails:
        f = open(email, errors= "ignore")
        blob = f.read()
        words += blob.split(" ")
        print(c)
        c -= 1

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]
    return dictionary.most_common(30000)


def make_dataset(dictionary):
    direc = "E:/Development work/SpamClassifier/enron1/emails/"
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    feature_set = []
    labels = []
    c = len(emails)

    for email in emails:
        data = []
        f = open(email, errors= "ignore")
        words = f.read().split(' ')
        for entry in dictionary:
            data.append(words.count(entry[0]))
        feature_set.append(data)

        if "ham" in email:
            labels.append(0)
        if "spam" in email:
            labels.append(1)
        print(c)
        c = c - 1
    return feature_set, labels


d = make_dict()
features, labels = make_dataset(d)

x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

clf = MultinomialNB(alpha = 0.0001)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("1")
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
save(clf, "Multinomial-text-classifier.mdl")

clf = ComplementNB(alpha = 0.0001)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("2")
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
save(clf, "Complement-text-classifier.mdl")

clf = GaussianNB()
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("3")
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
save(clf, "Gaussian-text-classifier.mdl")

clf = BernoulliNB(alpha = 0.0001)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("4")
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
save(clf, "Bernoulli-text-classifier.mdl")

clf = LinearSVC()
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("5")
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
save(clf, "LinearSVC-text-classifier.mdl")


clf = SVC()
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("6")
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
save(clf, "SVC-text-classifier.mdl")

