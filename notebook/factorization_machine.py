
def process(df, idxToLenId, links, features = ['year', 'votes', 'runtimes'])):
	"""Pre-process the movie's attribute and user's attributes to feature
        Args: 
            df (dataframe) : dataframe of movie attributes
        Returns: 
            UV (numpy 2D array): the user/movie features
    """



	