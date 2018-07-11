 
from base import BaseAlgo
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances
from cexc import get_logger
from util import df_util
from util.param_util import convert_params

# Everyone's favorite in memory collaborative filter, not a scaleable solution for millions of users and millions of items
# https://en.wikipedia.org/wiki/Collaborative_filtering
# please check out more scaleable solutions in KNN or "Recommender Systems: The Textbook"
# TODO add coldstart solution  for nulls
# TODO currently we assume a |fillnull value=0 is run in splunk prior to calling the algorithm

# We ASSUME rows are users, columns are items. 
# TODO I seem to cause splunk memory issues with wide tables, so I should consider doing an XYSERIES like reshape 
# TODO and consider taking in a table of USERID, ITEM , RATING from splunk. Yucky.

# TODO There are many many many other distance metrics that could be a good fit.


class CollaborativeFilter(BaseAlgo):
    def __init__(self, options):


       # set parameters
        params = options.get('params', {})
        out_params = convert_params(
            params,
            strs=['user_field','rating_type','coldstart_field']
        )

        # set defaults for parameters
        if 'user_field' in out_params:
            self.user_field = out_params['user_field']
        else:
            self.user_field = "SME"

        self.rating_type="item"
        if 'rating_type' in out_params:
            if out_params['rating_type'] == "item":
                self.rating_type="item"
            elif out_params['rating_type'] == "user":
                self.rating_type="user"


    def fit(self, df, options):
        # df contains all the search results, including hidden fields
        # but the requested requested are saved as self.feature_variables
        logger = get_logger('MyCustomLogging')

        X=df.copy()

        # it is always best practice to prepare your data.
        # splunk has a number of hidden fields that are exposed as part of the search protocole, and we really only
        # want the features that are valid field names.


        #Make sure to turn off get_dummies
        X, _, self.columns = df_util.prepare_features(
            X=X,
            variables=self.feature_variables,
            get_dummies=False,
            mlspl_limits=options.get('mlspl_limits'),
        )

        # test if user field is in the list
        logger.debug("The user field is %s",self.user_field )
        try: 
            my_list_index=(X[self.user_field].values)
        except:
            raise RuntimeError('You must specify user field that exists. You sent %s',self.user_field)

        X=X.drop([self.user_field],axis=1)
        my_list_header=(X.columns.values)

        #ratings as a matrix , clean that data up!
        X=X.replace([np.inf, -np.inf], "nan").replace("nan","0")
        matrix=X.values
        # force type for Numpy Math
        matrix=matrix.astype(np.float64)

        # should consider erroring out when you have super sparse user data
        # TODO add other methods via parameter
        user_sim = pairwise_distances(matrix, metric='cosine')
        item_sim = pairwise_distances(matrix.T, metric='cosine')

        #item prediction
        item_sim= matrix.dot(item_sim) / np.array([np.abs(item_sim).sum(axis=1)])

        #user sim
        mean_user_rating = matrix.mean(axis=1)
        matrix_diff = (matrix - mean_user_rating[:, np.newaxis])
        user_sim = mean_user_rating[:, np.newaxis] + user_sim.dot(matrix_diff) / np.array([np.abs(user_sim).sum(axis=1)]).T

        # add back into the matrix the header row
        if self.rating_type == "item":
            output_df=pd.DataFrame(item_sim,columns=my_list_header, index=my_list_index)
        if self.rating_type == "user":
            output_df=pd.DataFrame(user_sim,columns=my_list_header, index=my_list_index)        
        output_df[self.user_field]=pd.Series(my_list_index).values

        return output_df  
        

 

