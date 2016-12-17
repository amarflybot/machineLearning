# This is Naive based Sentiment Analysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


def main():
    with open("amazon_cells_labelled.txt", "r") as text_file:
        lines = text_file.read().split('\n')
    with open("imdb_labelled.txt", "r") as text_file:
        lines += text_file.read().split('\n')
    with open("yelp_labelled.txt", "r") as text_file:
        lines += text_file.read().split('\n')
    lines = [line.split("\t") for line in lines if len(line.split("\t")) == 2 and line.split("\t")[1] <> '']
    train_documents = [line[0] for line in lines]
    train_labels = [int(line[1]) for line in lines]
    count_vectorizer = CountVectorizer(binary='true')
    # Create Data Set
    train_documents = count_vectorizer.fit_transform(train_documents)
    # Training Phase
    classifier = BernoulliNB()
    classifier.fit(train_documents, train_labels)

    # Test Phase
    predict = classifier.predict(count_vectorizer.transform(["I hate you"]))
    print predict


if __name__ == "__main__": main()
