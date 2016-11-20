import numpy as np


class user_model_lr():
    """ Assume each user is a linear regression model. 
        The weights are solved by least square. 
        Input the pre-processed movie feature it will output rating.
    """
    def __init__(self, id):
        self.id = id # user's id
        self.w = None # user's weight

    def train(self, X, y):
        self.w = np.linalg.lstsq(X, y)[0]

    def predict(self, X):
        return X.dot(self.w)

def process(df):
    """ Pre-process the movie's attribute to vector of ['year', 'votes', 'runtimes']

        Args: 
            df (dataframe) : dataframe of movie attributes
            
        Returns: 
            movie_feature_dict (dict): each key map to a moive feature vector.
    """
    def movieToVec(df):
        vec = []
        vec += [df['year'], df['votes'], df['runtimes']]
        
        return vec


    # Get a subset
    df_movies_small = df[['year', 'votes', 'runtimes']]

    # Clean up data
    df_movies_small.loc[:,'year'] = map(parser, df_movies_small.loc[:,'year'])
    df_movies_small.loc[:,'year'] = df_movies_small.loc[:,'year'].fillna(df_movies_small.loc[:,'year'].mean())
    df_movies_small.loc[:,'votes'] = map(parser, df_movies_small.loc[:,'votes'])
    df_movies_small.loc[:,'votes'] = df_movies_small.loc[:,'votes'].fillna(df_movies_small.loc[:,'votes'].mean())
    df_movies_small.loc[:,'runtimes'] = map(parser, df_movies_small.loc[:,'runtimes'])
    df_movies_small.loc[:,'runtimes'] = df_movies_small.loc[:,'runtimes'].fillna(df_movies_small.loc[:,'runtimes'].mean())
    

    movie_feature_dict = {}
    for index, row in df_movies_small.iterrows():
        imdbId = index
        movie_feature_dict[imdbId] = movieToVec(row)
        
    return movie_feature_dict

def parser(x):
    if str(x).rstrip(']').lstrip('[').split()[0].rstrip(',') == 'nan':
        return np.nan
    else:
        return float(str(x).rstrip(']').lstrip('[').split()[0].rstrip(','))

    
def train(X_tr,  movie_feature_dict, idxToImdbId):
    """ Create and train the user models
        Args:
            X_tr (numpy 2D array) : a ratings matrix for training
            movie_feature_dict (dict) : a map to movie feature vector as the input of user_model
        Returns: 
            user_models (list) : a list of trained user_models
    """
    user_models = []
    
    for i, row in enumerate(X_tr):
        u = user_model_lr(i)
        X = np.array([movie_feature_dict[idxToImdbId[j]] for j in row.nonzero()[0]])

        y = np.array([X_tr[i, j] for j in row.nonzero()[0]])
        u.train(X, y)
        user_models.append(u)


    return user_models


def error(X, R):
    """ Compute the mean error of the observed ratings in X and their estimated values. 
        Args: 
            X (numpy 2D array) : a ratings matrix as specified above
            R (numpy 2D array) : reconstructed matrix
        Returns: 
            (float) : the mean squared error of the observed ratings with their estimated values
        """
    indicator = zip(X.nonzero()[0], X.nonzero()[1])
    
    error = 0.
    for i, j  in indicator:
        error += np.square(X[i,j] - R[i,j])
        
    error = error / float(np.count_nonzero(X))
    
    return error


def reconstruct(user_models, movie_feature_dict, idxToImdbId):
    """
    Reconstruct the rating matrix based on trained user model
    and movie feature.
        Args:
            user_models (list) : a list of user_model, where the parameter is trained
            movie_feature_dict (dict) : a map to movie feature vector as the input of user_model
            idxToImdbId (dict) : a map to from col index to imdbId
        Returns:
            X (numpy 2D array) : a reconstructed rating matrix
    """
    M = np.array([movie_feature_dict[idxToImdbId[j]] for j in range(len(idxToImdbId))])
    X = np.zeros((len(user_models), M.shape[0]))
    for i, u in enumerate(user_models):
        for j, m in enumerate(M):
            X[i, j] = u.predict(m)
    
    return X


# # Make sure the test_idxToLenId is correct
# def test_idxToLenId():
#     for i, name in enumerate(movieNames):
#         assert(movies[movies['movieId'] == idxToLenId[i]].title.values[0] is name)
    

# # Make sure thre is no missing movie 
# def test_movie():
#     for i in range(X_tr.shape[1]):
#         try:
#             df_imdb_movies.loc[idxToImdbId[i]]
#         except:
#             print idxToImdbId[i]
#             assert(False)
            
# test_idxToLenId()
# test_movie()

# # Visualization Training_data 
# fig = plt.figure(figsize=(18, 10))
# plt.imshow(X_tr, cmap = plt.cm.gray, interpolation="nearest")