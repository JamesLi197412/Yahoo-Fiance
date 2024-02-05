import numpy as np

class Exploration:
    def df_exploration(self, dataframe):
        """
            Information about the DataFrame
            # Columns Data Type
            # Data Frame shape
            # Columns Name
            # Columns Description
            # First 5 Data Samples
        """
        features_dtypes = dataframe.dtypes
        rows, columns = dataframe.shape

        missing_values_cols = dataframe.isnull().sum()
        missing_col = missing_values_cols.sort_values(ascending=False)
        features_names = missing_col.index.values
        missing_values = missing_col.values

        print('=' * 50)
        print('===> This data frame contains {} rows and {} columns'.format(rows, columns))
        print('=' * 50)

        print("{:13}{:13}{:30}".format('Feature Name'.upper(),
                                            'Data Format'.upper(),
                                            'The first five samples'.upper()))

        for features_names, features_dtypes in zip(features_names, features_dtypes[features_names]):
            print('{:15} {:14} '.format(features_names, str(features_dtypes)),end=" ")

            for i in range(5):
                print(dataframe[features_names].iloc[i], end=",")

            print("=" * 50)
