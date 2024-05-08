#!/usr/bin/python3
import numpy as np
from sklearn import decomposition

#Shift the original data onto axes defined by the principal components
#Augment the 3rd and 4th components by adding random value from a 0, 0.1 normal distribution
#Transform the data back to the original axes and return
def generateData(pca, x, start):
    original = pca.components_.copy()
    ncomp = pca.components_.shape[0]
    a = pca.transform(x)
    for i in range(start, ncomp):
        pca.components_[i,:] += np.random.normal(scale=0.1, size=ncomp)
    b = pca.inverse_transform(a)
    pca.components_ = original.copy()
    return b

def main():
    x = np.load("../data/iris/iris_features.npy")
    y = np.load("../data/iris/iris_labels.npy")

    #Separate into training and test data
    N = 120 
    x_train = x[:N]
    y_train = y[:N]
    x_test = x[N:]
    y_test = y[N:]

    #Initialize PCA and find component vectors for the data set
    pca = decomposition.PCA(n_components=4)
    pca.fit(x)
    print(pca.explained_variance_ratio_)

    #2 most significant components are untouched and create 9 new sets
    start = 2
    nsets = 10
    
    #Create empty arrays to store the new sets
    nsamp = x_train.shape[0]
    newx = np.zeros((nsets*nsamp, x_train.shape[1]))
    newy = np.zeros(nsets*nsamp, dtype="uint8")

    #Populate the first set with the original set and the subsequent 9 with augmented data
    for i in range(nsets):
        if (i == 0):
            newx[0:nsamp,:] = x_train
            newy[0:nsamp] = y_train

        else:
            newx[(i*nsamp):(i*nsamp+nsamp),:] = generateData(pca, x_train, start)
            newy[(i*nsamp):(i*nsamp+nsamp)] = y_train
    
    #Randomize the ordering of all data
    idx = np.argsort(np.random.random(nsets*nsamp))
    newx = newx[idx]
    newy = newy[idx]
    np.save("iris_train_features_augmented.npy", newx)
    np.save("iris_train_labels_augmented.npy", newy)
    np.save("iris_test_features_augmented.npy", x_test)
    np.save("iris_test_labels_augmented.npy", y_test)

if __name__ == "__main__":
    main()