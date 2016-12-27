from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def main():
    with open("../naiveBayes/imdb_labelled.txt","r") as text_file:
        lines = text_file.read().split('\n')
    lines = [line.split("\t") for line in lines if len(line.split("\t")) == 2 and line.split("\t")[1] != '']
    train_documents = [line[0] for line in lines]
    tfid_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
    train_documents = tfid_vectorizer.fit_transform(train_documents)
    k_means = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=True)
    means_fit = k_means.fit(train_documents)
    count = 0
    for i in range(len(lines)):
        if count > 3:
            break
        if means_fit.labels_[i] == 0:
            print(lines[i])
            count+=1


if __name__ == '__main__':
    main()