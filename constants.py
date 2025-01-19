CLUSTER = 'cluster'
BINS = 'bin'
HIST = 'HIST'
DATA_TYPE = 'data_type'
NUMERIC_TYPE = 'numeric'
CATEGORICAL_TYPE = 'categorical'
BINNED_SUFFIX = '_BINNED'
PROBA_COL = 'proba'
LIFT_SUFFIX = '_LIFT'
COHORT = 'cohort'
DIST = 'dist'
INTEREST = 'interest'
DIVERS = 'diversity'
SUFF = 'suff'
SINGLE = 'single'

COMBINATIONS = {
        # 'Adult': {
        #     # 'Agglomerative': ['onehot', 'VIME', 'LabelEncoder', 'FreqEncoder', 'TargetEncoder'],
        #     'Agglomerative': ['LabelEncoder'],
        #     # 'GaussianMixture': ['onehot', 'VIME', 'LabelEncoder', 'FreqEncoder', 'TargetEncoder'],
        #     'GaussianMixture': ['LabelEncoder'],
        #     'kmeans': [ 'LabelEncoder'],
        #     # 'kmeans': ['onehot', 'VIME', 'LabelEncoder', 'FreqEncoder', 'TargetEncoder'],
        #     'diffpriv_kmeans': ['LabelEncoder'],
        #     'kmodes': ['none'],
        #     # 'labels': ['none']
        # },
        # 'cdc_diabetes': {
        #     # 'GaussianMixture': ['onehot', 'VIME', 'LabelEncoder', 'FreqEncoder', 'TargetEncoder', 'none_scaled'],
        #     'GaussianMixture': ['none_scaled'],
        #     'kmeans': ['none_scaled'],
        #     'diffpriv_kmeans': ['none_scaled'],
        #     'kmodes': ['none'],
        #     # 'labels': ['none']
        # },
        # 'cdc_diabetes_sample': {
        #      'Agglomerative': ['none_scaled'],
        #     'GaussianMixture': ['none_scaled'],
        #     'kmeans': ['none_scaled'],
        #     'diffpriv_kmeans': ['none_scaled'],
        #     'kmodes': ['none'],
        # },
          'census_1990': {
            # 'kmeans': ['onehot', 'VIME', 'LabelEncoder', 'FreqEncoder', 'none_scaled'],
            'kmeans': ['none_scaled'],
            'diffpriv_kmeans': ['none_scaled'],
            'kmodes': ['none'],
            # 'labels': ['none'],
            'GaussianMixture': ['none']
        },
        # 'census_1990_sample': {
        #     'Agglomerative': ['none_scaled'],
        #     'kmeans': ['none_scaled'],
        #     'diffpriv_kmeans': ['none'],
        #     'kmodes': ['none'],
        #     # 'labels': ['none'],
        #     'GaussianMixture': ['none']
        # },
        # 'Nursery': {
        #     'Agglomerative': [ 'LabelEncoder'],
        #     # 'Agglomerative': ['onehot', 'VIME', 'LabelEncoder', 'FreqEncoder', 'TargetEncoder'],
        #     'GaussianMixture': ['LabelEncoder'],
        #     # 'GaussianMixture': ['onehot', 'VIME', 'LabelEncoder', 'FreqEncoder', 'TargetEncoder'],
        #     # 'kmeans': ['onehot', 'VIME', 'LabelEncoder', 'FreqEncoder', 'TargetEncoder'],
        #     'kmeans': [ 'LabelEncoder'],
        #     'kmodes': ['none'],
        #     # 'labels': ['none'],
        #     'diffpriv_kmeans': ['LabelEncoder']
        # },
    # ,
    #     'car_accidents': {
    #     'Agglomerative': ['LabelEncoder'],
    #     'GaussianMixture': ['LabelEncoder'],
    #     'kmeans': ['LabelEncoder'],
    #     'kmodes': ['none'],
    #     'labels': ['none'],
    #     'diffpriv_kmeans': ['LabelEncoder']
    # },
        # 'netflix': {
        #     'kmeans': ['none'],
        #     'diffpriv_kmeans': ['none']
        # }

    # 'COIL': {
    #     'GaussianMixture': ['none'],
    #     'kmeans': ['none'],
    #     'diffpriv_kmeans': ['none'],
    #     'kmodes': ['none'],
    #     # 'labels': ['none'],
    #     'Agglomerative': ['none']
    #     },
    # 'SpeedDate': {
    #     'GaussianMixture': ['none_scaled'],
    #     'kmeans': ['none_scaled'],
    #     'diffpriv_kmeans': ['none'],  ## 2 creates only one cluster
    #     'kmodes': ['none'],
    #     # 'labels': ['none'],
    #     'Agglomerative': ['none_scaled']
    #     },

    # 'MSWEB': {
    #     # 'GaussianMixture': ['none_scaled'],
    #     # 'kmeans': ['none_scaled'],
    #     # 'diffpriv_kmeans': ['none'],
    #     'kmodes': ['none'],
    #     # 'labels': ['none'],
    #     'Agglomerative': ['none']
    # },

    # 'Amazon': {
    #     'GaussianMixture': ['none'],
    #     'kmeans': ['none'],
    #     # 'diffpriv_kmeans': ['none'],
    #     # 'kmodes': ['none'],
    #     # 'labels': ['none'],
    #     'Agglomerative': ['none']
    # },

    # 'Parkinsons': {
    #     'GaussianMixture': ['none_scaled'],
    #     'kmeans': ['none_scaled'],
    #     'diffpriv_kmeans': ['none'],
    #     'kmodes': ['none'],
    #     # 'labels': ['none'],
    #     'Agglomerative': ['none']
    # },
    # 'Student': {
    #     'GaussianMixture': ['none_scaled'],
    #     'kmeans': ['none_scaled'],
    #     'diffpriv_kmeans': ['none'],
    #     'kmodes': ['none'],
    #     # 'labels': ['none'],
    #     'Agglomerative': ['none']
    # },

    'Diabetes2': {
            'Agglomerative': ['LabelEncoder'],
            'GaussianMixture': ['LabelEncoder'],
            'kmeans': [ 'LabelEncoder'],
            'kmodes': ['none'],
            'diffpriv_kmeans': ['LabelEncoder']
        },

}