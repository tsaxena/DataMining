# Event
# Hackbright Data Science Workshop.

# Author
# Daniel Wiesenthal.  dw@cs.stanford.edu.

# What is this?
# This is a simple script illustrating the usage of the Python NLTK classifier.  It is written in Python, but the comments are intended to make it clear how to port to other languages.  The flow isn't particularly well decomposed as a program; rather, it is intended to go along linearly with the associated talk/presentation.
# The goal is to find out which chocolate a particular volunteer during the talk will like.  We have a few examples of chocolate bars that we know are either matches or not (misses), and want to use that to make a guess about an unknown bar (we don't know if it will be a match, and want to guess).

# Further reading:
# http://www.stanford.edu/class/cs124/lec/naivebayes.pdf
# http://nltk.googlecode.com/svn/trunk/doc/book/ch06.html

# Software Setup
# For this script to work, you'll need to have Python, NLTK (a Python package), and Numpy (upon which NLTK depends) installed.  On a Mac (which all have numpy pre-installed these days), run:
# sudo easy_install pip
# sudo pip install nltk
# <cd to directory with this file>
# python classification_101.py

#Required libraries
try:
    import nltk
    from nltk.classify.util import apply_features
    import string
    print "Great!  Looks like you're all set re: NLTK and Python."
except Exception, e:
    print "Bummer.  Looks like you don't have NLTK and Python set up correctly.  (Exception: "+str(e)+")"
    quit()
raw_input("\n\nHit enter to get started...")


#Some example chocolate bars.  The format is a tuple of {information about the chocolate bar} and a {value}, where "match" is a good match and "miss" is a poor/bad match.
print "Defining Training Data (hand-coded in this case, see script source)"
known_1 = ("fruity dark organic sweet chocolate", "miss")
known_2 = ("interesting spicy dark bitter", "miss")
known_3 = ("sweet caramel crunchy light salty", "match")
known_4 = ("fruity dark organic bitter", "miss")
known_5 = ("sweet dark crunchy bitter interesting fruity", "match")
known_6 = ("light milky sweet", "match")
known_7 = ("refreshing dark sweet minty", "match")
known_8 = ("dark organic bitter", "miss")
known_9 = ("dark bitter bitter plain intense ghirardelli scary", "miss")
known_10 = ("organic dark salty bitter", "miss")

known_data_points = [known_1, known_2, known_3, known_4, known_5, known_6, known_7, known_8, known_9, known_10]


raw_input("\n\nHit enter to continue...")
#Feature extractor.  Basically takes a sentence/phrase/description/whatever and and outputs a stripped version of it.  This could/should be enhanced (with, eg, stemming), as that would provide easy gains in performance, but this is a good start.  Once you get the basic flow set up for a classification project, you'll spend most of your time in feature extraction.
print "Writing Feature Extractor"
def feature_extracting_function(data_point):
    features = {} #Dictionary, roughly equivalent to a hashtable in other languages.
    data_point = ''.join(ch for ch in data_point if ch not in set(string.punctuation)) #Strip punctuation characters from the string. In Python, this happens to be usually done with a .join on the string object, but don't be thrown if you're used to other languages and this looks weird (hell, it looks weird to me), all we're doing is stripping punctuation.
    words = data_point.split() #Split data_point on whitespace, return as list
    words = [word.lower() for word in words] #Convert all words in list to lowercase.  The [] syntax is a Python "list comprehension"; Google that phrase if you're confused.

    #Create a dictionary of features (True for each feature present, implicit False for absent features).  In this case, features are words, but they could be bigger or smaller, simpler or more complex.
    for word in words:
        features["contains_word_(%s)" % word] = True
    return features


raw_input("\n\nHit enter to continue...")
print "Extracting Features from Training Set"
train_set = apply_features(feature_extracting_function, known_data_points)


raw_input("\n\nHit enter to continue...")
print "Gathering unknown data points (new data) to predict on (again, hand-coded, see script source)"
#Our query chocolate bars: we want to know whether or not they're matches
unknown_1 = "milky light sweet nutty"
unknown_2 = "dark bitter plain"
unknown_3 = "dark dark bitter beyond belief organic"
unknown_4 = "organic minty sweet dark"


raw_input("\n\nHit enter to continue...")
#Train a Naive Bayes Classifier (simple but surprisingly effective).  This isn't the only classifier one could use (dtree is another, and there are many, many more), but it's a good start.
print "Training Naive Bayes Classifier"
nb = nltk.NaiveBayesClassifier.train(train_set)


raw_input("\n\nHit enter to continue...")
#Make guesses about our unknown projects:
print "Predicting the class of unknown data points"
print "Prediction for unknown_1: "+str(nb.classify(feature_extracting_function(unknown_1)))
print "Prediction for unknown_2: "+str(nb.classify(feature_extracting_function(unknown_2)))
print "Prediction for unknown_3: "+str(nb.classify(feature_extracting_function(unknown_3)))
print "Prediction for unknown_4: "+str(nb.classify(feature_extracting_function(unknown_4)))


raw_input("\n\nHit enter to continue...")
#Now get some insight as to how well the classifier performs in general.  The right way to do this is to have a test set of examples that were not used to train the classifier, because otherwise you're just asking for a false sense of confidence (it will report that it does very well--well, of course!  Of course it's gonna do well on the things you trained it on--what you want to see is whether it can handle new data or not).  Read more about test, train, and validation sets to do it better.  Google "10 fold cross validation" to get started on really doing it right.
test_set = train_set #No no no no no.  Except for illustration cases like in a skillshare script, then yes. :P
print "Evaluating Accuracy on Training Set (WARNING! This is just for illustration purposes, don't use train set for evaluation in practice!)"
print "Accuracy: "+str(nltk.classify.accuracy(nb, test_set))


raw_input("\n\nHit enter to continue...")
#Print the features that are most influential in making the decision of whether it's a good match or not.  Note that many of the features are presented in a format where the feature being "None" is meaningful; this is basically meant to be read as "When contains_word_(jams) is false/none, then that matters this much..."  See the nltk page referenced above for more info.
print "Let's look deeper into the classifier..."
print str(nb.show_most_informative_features(20))


raw_input("\n\nHit enter to continue...")
#Another interesting classifier, which can print out pseudocode for making a decision (just included in one line for fun).
print "Let's explore another (not NB) classifier, Decision Tree.  Because of the inherent structure of a Decision Tree classifier, we can print it out as a series of decisions made in pseudocode."
print nltk.DecisionTreeClassifier.train(train_set).pseudocode(depth=5)
