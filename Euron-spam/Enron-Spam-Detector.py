# Imports
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import random

# Enron Dataset Location
rootdir = "data"

# Loop through all the directories, sub directories and files in the above folder, and print them.
# For files, print number of files.
for directories, subdirs, files in os.walk(rootdir):
    print(directories, subdirs, len(files))

print(os.path.split("data\\enron1\\ham"))
print(os.path.split("data\\enron1\\ham")[0])
print(os.path.split("data\\enron1\\ham")[1])

# Same as before, but only print the ham and spam folders
for directories, subdirs, files in os.walk(rootdir):
    if (os.path.split(directories)[1]  == 'ham'):
        print(directories, subdirs, len(files))
    
    if (os.path.split(directories)[1]  == 'spam'):
        print(directories, subdirs, len(files))

ham_list = []
spam_list = []

# Same as before, but this time, read the files, and append them to the ham and spam list
for directories, subdirs, files in os.walk(rootdir):
    if (os.path.split(directories)[1]  == 'ham'):
        for filename in files:      
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                ham_list.append(data)
    
    if (os.path.split(directories)[1]  == 'spam'):
        for filename in files:
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                spam_list.append(data)


print(ham_list[0])
print('--------- ')
print(spam_list[0])

# Write a function , that when passed in words, will return a dictionary of the form
# {Word1: True, Word2: True, Words3: True}
# Removing stop words is optional

def create_word_features(words):
    my_dict = dict( [ (word, True) for word in words] )
    return my_dict

# Test the function
create_word_features(["the", "quick", "brown", "quick", "a", "fox"])

ham_list = []
spam_list = []

# Same as before, but this time:

# 1. Break the sentences into words using word_tokenize
# 2. Use the create_word_features() function you just wrote
for directories, subdirs, files in os.walk(rootdir):
    if (os.path.split(directories)[1]  == 'ham'):
        for filename in files:      
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                
                # The data we read is one big string. We need to break it into words.
                words = word_tokenize(data)
                
                ham_list.append((create_word_features(words), "ham"))
    
    if (os.path.split(directories)[1]  == 'spam'):
        for filename in files:
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                
                # The data we read is one big string. We need to break it into words.
                words = word_tokenize(data)
                
                spam_list.append((create_word_features(words), "spam"))
print(ham_list[0])
print(spam_list[0])

combined_list = ham_list + spam_list
print(len(combined_list))

random.shuffle(combined_list)

# Create a test and train section.
# 60% of the data is training. 40% is test

training_part = int(len(combined_list) * .6)

print(len(combined_list))

training_set = combined_list[:training_part]

test_set =  combined_list[training_part:]

print (len(training_set))
print (len(test_set))

# Create the Naive Bayes filter
classifier = NaiveBayesClassifier.train(training_set)

# Find the accuracy, using the test data
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print("Accuracy is: ", accuracy * 100)

classifier.show_most_informative_features(20)

# Clasify the below as spam or ham
# 1. Break into words using word_tokenzise
# 2. create_word_features
# 3. Use the classify function

msg1 = '''Hello th̓ere seُx master :-)
i need c0ck ri͏ght noِw ..͏. don't tell my hǔbbٚy.ٚ. ))
My sc͕rٞeٚe̻nname is Dorry.
My accֺo֔unt is h֯ere: http:nxusxbnd.GirlsBadoo.ru
C u late٘r!'''


msg2 = '''As one of our top customers we are providing 10% OFF the total of your next used book purchase from www.letthestoriesliveon.com. Please use the promotional code, TOPTENOFF at checkout. Limited to 1 use per customer. All books have free shipping within the contiguous 48 United States and there is no minimum purchase.

We have millions of used books in stock that are up to 90% off MRSP and add tens of thousands of new items every day. Don’t forget to check back frequently for new arrivals.'''



msg3 = '''To start off, I have a 6 new videos + transcripts in the members section. In it, we analyse the Enron email dataset, half a million files, spread over 2.5GB. It's about 1.5 hours of  video.

I have also created a Conda environment for running the code (both free and member lessons). This is to ensure everyone is running the same version of libraries, preventing the Works on my machine problems. If you get a second, do you mind trying it here?'''

words = word_tokenize(msg1)
features = create_word_features(words)
print("Message 1 is :" ,classifier.classify(features))

words = word_tokenize(msg2)
features = create_word_features(words)
print("Message 2 is :" ,classifier.classify(features))

words = word_tokenize(msg3)
features = create_word_features(words)
print("Message 3 is :" ,classifier.classify(features))


