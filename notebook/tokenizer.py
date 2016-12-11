def tokenize(jointTable):
    data, y = [], []
    for index, row in jointTable.iterrows():

        feature_dict = {}

        # # Genres
        # for cate in row['genres'].split(','):
        #     key = 'category={}'.format(cate)
        #     feature_dict[key] = 1
            
        # # Occupation
        # for cate in row['occupation'].split(','):
        #     key = 'occupation={}'.format(cate)
        #     feature_dict[key] = 1

        # # Age 
        # feature_dict['age'] = float(row['age'])
        
        # # Release year
        # feature_dict['release_year'] = float(row['year'])
        
        # # Votes
        # feature_dict['votes'] = float(row['votes'])
        
        # # Gender
        # feature_dict['gender'] = str(row['gender'])
        
        # ImdbId
        feature_dict['imdbId'] = str(row['imdbId'])
        
        # UserId
        feature_dict['userId'] = str(row['userId'])

        data.append(feature_dict)
        y.append(float(row['rating']))

    return data, y

            
