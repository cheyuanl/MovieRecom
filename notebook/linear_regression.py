import numpy as np

def process(df, idxToLenId, links, features = ['year', 'votes', 'runtimes']):
    """ Pre-process the movie's attribute to vector of selected attributes
        Args: 
            df (dataframe) : dataframe of movie attributes
        Returns: 
            V (numpy 2D array): the movie features
    """
    links = links.set_index('movieId')
    df = df.set_index('imdbId')
    # TODO: make it cleaner...

    def movieToVec(series, features):        
        return np.append(series[features].values.astype(float), 1.) ## 1 is the bias term

    def numeric_parser(x):
        if str(x).rstrip(']').lstrip('[').split()[0].rstrip(',') == 'nan':
            return np.nan
        else:
            return float(str(x).rstrip(']').lstrip('[').split()[0].rstrip(','))

    # def list_parser(x):
    #     print repr(x)
    #     return x
    
    # Clean up data
    df.loc[:,'year'] = map(numeric_parser, df.loc[:,'year'])
    df.loc[:,'year'] = df.loc[:,'year'].fillna(df.loc[:,'year'].mean())
    df.loc[:,'votes'] = map(numeric_parser, df.loc[:,'votes'])
    df.loc[:,'votes'] = df.loc[:,'votes'].fillna(df.loc[:,'votes'].mean())
    df.loc[:,'runtimes'] = map(numeric_parser, df.loc[:,'runtimes'])
    df.loc[:,'runtimes'] = df.loc[:,'runtimes'].fillna(df.loc[:,'runtimes'].mean())
    # df.loc[:,'genres'] =  map(list_parser, df.loc[:,'genres'])

    # genres = set(reduce(lambda x, y : x + y, df.loc[:,'genres']))
    # genres = [i.split(',')[0] for i in genres if i != 'nan']
    # print len(genres)
    # print genres

    
    idxToImdbId = [int(links.loc[i].imdbId) for i in idxToLenId]
    return np.array([movieToVec(df.loc[idxToImdbId[i], :], features) for i in range(len(idxToLenId))]), idxToImdbId

    
def train(X_tr, V, lam = 1):
    """ Create and train the user models
        Args:
            X_tr (numpy 2D array) : ratings matrix for training
            V (numpy 2D array) : movie attribute vectors
        Returns: 
            U (numpy 2D array) : the user parameters
    """

    U = np.zeros((X_tr.shape[0], V.shape[1]))
    for i, row in enumerate(X_tr):
        nzs = X_tr[i,:].nonzero()[0]
        vv = V[nzs,:]        

        y = np.array([X_tr[i, j] for j in row.nonzero()[0]])
        # U[i, :] = np.linalg.lstsq(vv, y)[0]
        U[i,:] = np.linalg.inv(vv.T.dot(vv) + lam * np.eye(V.shape[1])).dot(vv.T.dot(X_tr[i,nzs]).T)

    return U








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