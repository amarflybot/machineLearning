from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def main():
    with open("../naiveBayes/imdb_labelled.txt","r") as text_file:
        lines = text_file.read().split('\n')
    lines = [line.split("\t") for line in lines if len(line.split("\t")) == 2 and line.split("\t")[1] != '']
    train_documents = [line[0] for line in lines]
    tfid_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
    train_documents = tfid_vectorizer.fit_transform(train_documents)
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=True)
    means_fit = kmeans.fit(train_documents)
    count = 0
    for i in range(len(lines)):
        if count > 3:
            break
        if kmeans.labels_[i] == 0:
            print(lines[i])
            count+=1
    
    # # Step size of the mesh. Decrease to increase the quality of the VQ.
    # h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    #
    # # Plot the decision boundary. For that, we will assign a color to each
    # x_min, x_max = train_documents[:, 0].min() - 1, train_documents[:, 0].max() + 1
    # y_min, y_max = train_documents[:, 1].min() - 1, train_documents[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # # Obtain labels for each point in mesh. Use last trained model.
    # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    #
    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(Z, interpolation='nearest',
    #            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #            cmap=plt.cm.Paired,
    #            aspect='auto', origin='lower')
    #
    # plt.plot(train_documents[:, 0], train_documents[:, 1], 'k.', markersize=2)
    # # Plot the centroids as a white X
    # centroids = kmeans.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1],
    #             marker='x', s=169, linewidths=3,
    #             color='w', zorder=10)
    # plt.title('K-means clustering on IMDB movies')
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()

if __name__ == '__main__':
    main()