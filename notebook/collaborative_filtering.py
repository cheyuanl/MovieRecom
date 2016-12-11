import pandas as pd
import numpy as np
import math


def process(ratings, movies, P):
    """ Given a dataframe of ratings and a random permutation, split the data into a training 
        and a testing set, in matrix form. 
        
        Args: 
            ratings (dataframe) : dataframe of MovieLens ratings
            movies (dataframe) : dataframe of MovieLens movies
            P (numpy 1D array) : random permutation vector
            
        Returns: 
            (X_tr, X_te, movie_names)  : training and testing splits of the ratings matrix (both 
                                         numpy 2D arrays), and a python list of movie names 
                                         corresponding to the columns of the ratings matrices. 
    """
    n = len(ratings)
    num_user = len(set(ratings['userId']))
    # assert (num_user == list(ratings['userId'])[-1])
    num_movie = len(set(movies['movieId']))
    # assert (num_movie == len(movies['movieId']))

    # Compress the movie index 
    table = list(set(ratings['movieId']))
    movieIdToIndex = {}
    idxToLenId = []
    for index, movieId in enumerate(table):
        movieIdToIndex[movieId] = index
        idxToLenId.append(movieId)
    
    ratings_tr = ratings.iloc[P[:int(math.floor(9*n /10.))]]
    ratings_te = ratings.iloc[P[int(math.floor(9*n /10.)):]]
        
    X_tr = np.zeros((num_user, len(movieIdToIndex)))
    X_te = np.zeros((num_user, len(movieIdToIndex)))
    # assert(len(ratings_tr.index) + len(ratings_te.index) == len(ratings.index))
    
    for row, item in ratings_tr.iterrows():
        X_tr[int(item['userId']-1), movieIdToIndex[item['movieId']]] = item['rating']

    for row, item in ratings_te.iterrows():
        X_te[int(item['userId']-1), movieIdToIndex[item['movieId']]] = item['rating']
        
    movie_names = [''] * len(movieIdToIndex)
    for movieId in table:
        movie_names[movieIdToIndex[movieId]] = movies.loc[movies['movieId'] == movieId]['title'].values[0]

    idxToUserId = range(1, num_user+1)
        
    
    return X_tr, X_te, movie_names, idxToUserId, idxToLenId

def error(X, U, V):
    """ Compute the mean error of the observed ratings in X and their estimated values. 
        Args: 
            X (numpy 2D array) : a ratings matrix as specified above
            U (numpy 2D array) : a matrix of features for each user
            V (numpy 2D array) : a matrix of features for each movie
        Returns: 
            (float) : the mean squared error of the observed ratings with their estimated values
        """
    indicator = zip(X.nonzero()[0], X.nonzero()[1])
    R = U.dot(V.T)
    
    error = 0
    for i, j  in indicator:
        error += np.square(X[i,j] - R[i,j])
        
    error = error / float(np.count_nonzero(X))
    
    return error

def train(X, X_te, k = 5, U = None, V = None, niters=51, lam=10, verbose=False):
    """ Train a collaborative filtering model. 
        Args: 
            X (numpy 2D array) : the training ratings matrix as specified above
            k (int) : the number of features use in the CF model
            U (numpy 2D array) : an initial matrix of features for each user
            V (numpy 2D array) : an initial matrix of features for each movie
            niters (int) : number of iterations to run
            lam (float) : regularization parameter
            verbose (boolean) : verbosity flag for printing useful messages
            
        Returns:
            (U,V) : A pair of the resulting learned matrix factorization
    """    
    m, n = X.shape
    # initialization
    if U is None or V is None:
        U = np.random.rand(m, k)
        V = np.random.rand(n, k)

    W = np.zeros((m, n))
    indicator = zip(X.nonzero()[0], X.nonzero()[1])
    for i, j in indicator:
        W[i,j] = 1
        
    for it in xrange(niters):
        
        for i in xrange(m):
            nzs = W[i,:].nonzero()[0]
            vv = V[nzs,:]        
            # U[i,:] = np.linalg.inv(vv.T.dot(vv) + lam * np.eye(k)).dot(V.T.dot(X[i,:]).T)
            U[i,:] = np.linalg.inv(vv.T.dot(vv) + lam * np.eye(k)).dot(vv.T.dot(X[i,nzs]).T)
        
        for j in xrange(n):
            nzs = W[:,j].nonzero()[0]
            uu = U[nzs,:]
            # V[j,:] = np.linalg.inv(uu.T.dot(uu) + lam * np.eye(k)).dot(U.T.dot(X[:,j]))
            V[j,:] = np.linalg.inv(uu.T.dot(uu) + lam * np.eye(k)).dot(uu.T.dot(X[nzs,j]))
        if verbose and (it % 5 == 0):
            print "Iter: {:03d}\ttrain error: {:0.3f}\tTest error: {:0.4f}".format(it, error(X, U, V), error(X_te, U, V))
        

    return U, V

def recommend(X, U, V, movieNames):
    """ Recommend a new movie for every user.
        Args: 
            X (numpy 2D array) : the training ratings matrix as specified above
            U (numpy 2D array) : a learned matrix of features for each user
            V (numpy 2D array) : a learned matrix of features for each movie
            movieNames : a list of movie names corresponding to the columns of the ratings matrix
        Returns
            (list) : a list of movie names recommended for each user
    """
    prediction = U.dot(V.T)
    
    recommendation = []
    
    
    for i, user in enumerate(X):
        ratings = prediction[i,:]
        indices = sorted(range(len(ratings)), key=lambda k: ratings[k], reverse = True)

        for j in indices:
            if(user[j] == 0):
                recommendation.append(movieNames[j])
                break
    
    return 

# # Visualize the U V matrix
# fig = plt.figure(figsize=(18, 7))
# grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
#                  nrows_ncols=(1,3),
#                  axes_pad=0.15,
#                  share_all=True,
#                  cbar_location="right",
#                  cbar_mode="single",
#                  cbar_size="7%",
#                  cbar_pad=0.15,
#                  )

# grid[0].imshow(U[:20,:], cmap=plt.cm.gray, interpolation="nearest")
# grid[1].imshow(V[:20,:].T, cmap=plt.cm.gray, interpolation="nearest")
# grid[2].imshow(U[:20,:].dot(V[:20,:].T), cmap=plt.cm.gray, interpolation="nearest")
