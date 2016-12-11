import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, csc_matrix

def process (RatingMatrix, df_movie_features, df_user_features, movie_feature, user_feature, idxToUserId, idxToImdbId):
	""" Create the input data for factorization machine 
		Args:
			RatingMatrix (dataframe) : rating matrix (row: user, columns: item)
			df_movie_features (dataframe) a dataframe stores movie feature vectors
			df_user_features (dataframe) : a dataframe stores user feature vectors
			movie_feature (list) : selected movie attributes
			user_feature (list) : selected user attributes
		Returns: 
			X (2d sparse matrix) input data for FM
			y (numpy array) : target label)

	"""
	X, y = [], [] 
	count = 0
	for i, row in enumerate(RatingMatrix):
		for j in row.nonzero()[0]:
			u = df_user_features.loc[idxToUserId[i], user_feature][0]
			for uu in df_user_features.loc[idxToUserId[i], user_feature][1:]:
				u = hstack(u, uu, format = 'csc')
			v = df_movie_features.loc[idxToImdbId[j], movie_feature][0]
			for vv in df_movie_features.loc[idxToImdbId[j], movie_feature][1:]:
				v = hstack(v, vv, format = 'csc')

			X.append(hstack((u, v)))
			y.append(RatingMatrix[i,j])
			count += 1
			if(count % 100 == 0):
				print count


	return csr_matrix(X), np.array(y)

