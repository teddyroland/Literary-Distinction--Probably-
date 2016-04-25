import os
from nltk import word_tokenize
from collections import Counter
from nltk import NaiveBayesClassifier



### Import and tokenize volumes of poetry

random_path = 'poems/random/'
random_files = os.listdir(random_path)
random_volumes = [open(random_path+name,'r').read() for name in random_files]
random_tokenized = [word_tokenize(volume.lower()) for volume in random_volumes]

review_path = 'poems/reviewed/'
review_files = os.listdir(review_path)
review_volumes = [open(review_path+name,'r').read() for name in review_files]
review_tokenized = [word_tokenize(volume.lower()) for volume in review_volumes]



### Create a list of most common words across entire corpus

all_words = [token for volume in (random_tokenized + review_tokenized) for token in volume]
all_words_counted = Counter(all_words)

common_words_counts = all_words_counted.most_common(500)
common_words = [word for word,count in common_words_counts]



### Get our texts into the format NLTK expects for its classifier

random_featurized = [{word:word in volume for word in common_words} for volume in random_tokenized]
review_featurized = [{word:word in volume for word in common_words} for volume in review_tokenized]

random_tagged = [(volume,'random') for volume in random_featurized]
review_tagged = [(volume,'reviewed') for volume in review_featurized]

all_tagged = random_tagged + review_tagged



### Train the classifier

classifier = NaiveBayesClassifier.train(all_tagged)



### Import, tokenize, featurize new volumes of poetry

path = 'poems/canonic/'
canonic_files = os.listdir(path)
canonic_volumes = [open(path+file).read() for file in canonic_files]
canonic_tokenized = [word_tokenize(volume.lower()) for volume in canonic_volumes]
canonic_featurized = [{word:word in volume for word in common_words} for volume in canonic_tokenized]



### Predict whether these volumes might have been reviewed

print(classifier.classify_many(canonic_featurized))
print(classifier.show_most_informative_features(10))
