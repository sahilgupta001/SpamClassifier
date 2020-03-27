import pickle as c
import os
from pip._vendor.distlib.compat import raw_input
from collections import Counter


def load(clf_file):
    with open(clf_file, 'rb') as fp:
        clf = c.load(fp)
    return clf


def make_dict():
    direc = "E:/Development work/SpamClassifier/enron1/emails/"
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    words = []
    c = len(emails)

    for email in emails:
        f = open(email, errors = "ignore")
        blob = f.read()
        words += blob.split(" ")
        print(c)
        c -= 1

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]
    return dictionary.most_common(3000)


d = make_dict()

print("The types of algorithms supported are:- ")
print("1. Multinomial Naive Bayes")
print("2. Complement Naive Bayes")
print("3. Gaussian Naive Bayes")
print("4. Bernoulli Naive Bayes")
print("5. Linear SVC")
print("6. SVC")
print("7. To Exit the program")
while True:
    choice = int(input("Enter the number of the algorithm you want to use for prediction: "));
    features = []
    if (choice == 1):
        clf = load("Multinomial-text-classifier.mdl")
    elif (choice == 2):
        clf = load("Complement-text-classifier.mdl")
    elif (choice == 3):
        clf = load("Gaussian-text-classifier.mdl")
    elif (choice == 4):
        clf = load("Bernoulli-text-classifier.mdl")
    elif (choice == 5):
        clf = load("LinearSVC-text-classifier.mdl")
    elif (choice == 6):
        clf = load("SVC-text-classifier.mdl")
    elif (choice == 7):
        break
    print ("Enter the Email message: ")
    inp = raw_input(">").split()
    for word in d:
        features.append(inp.count(word[0]))
    res = clf.predict([features])
    print(["Not Spam", "Spam!"][res[0]])