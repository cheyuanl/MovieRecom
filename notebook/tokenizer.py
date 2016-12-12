def tokenize(jointTable, features = ['all']):
    data, y = [], []
    if 'all' in features:
        features = ['genres', 'occupation', 'age', 'release_year', 'votes', 'age', 'imdbId', 'userId']

    for index, row in jointTable.iterrows():
        feature_dict = {}
        if 'genres' in features:
            for cate in row['genres'].split(','):
                key = 'category={}'.format(cate)
                feature_dict[key] = 1
            
        if 'occupation' in features:
            for cate in row['occupation'].split(','):
                key = 'occupation={}'.format(cate)
                feature_dict[key] = 1

        if 'age' in features:
            feature_dict['age'] = float(row['age'])

        if 'release_year' in features:
            feature_dict['release_year'] = float(row['year'])
        
        if 'votes' in features:
            feature_dict['votes'] = float(row['votes'])
        
        if 'age' in features:
            feature_dict['gender'] = str(row['gender'])
        
        if 'imdbId' in features:
            feature_dict['imdbId'] = str(row['imdbId'])

        if 'userId' in features:
            feature_dict['userId'] = str(row['userId'])

        data.append(feature_dict)
        y.append(float(row['rating']))

    return data, y

            
