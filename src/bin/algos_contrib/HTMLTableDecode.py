from base import BaseAlgo
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances
from cexc import get_logger
from util import df_util
from bs4 import BeautifulSoup
import html5lib
import copy
import math
from util.param_util import convert_params


class HTMLTableDecode(BaseAlgo):
    def __init__(self, options):
        feature_variables = options.get('feature_variables', {})

        if len(feature_variables) == 0:
            raise RuntimeError('You must supply one or more fields')


    def fit(self, df, options):

        #make a copy of the original data
        X = df.copy()

        #get the list of header values
        my_list_header = (X.columns.values)

        #get the length of the data
        my_list_index = (X["content"].values)

        # add back into the matrix the header row
        output_df = pd.DataFrame()

        url_soup = BeautifulSoup(X["content"][0])
        headerrow=X["headerrow"][0]
        tables = []
        tables_html = url_soup.find_all("table")

        # Parse each table
        for n in range(0, len(tables_html)):

            n_cols = 0
            n_rows = 0

            for row in tables_html[n].find_all("tr"):
                col_tags = row.find_all(["td", "th"])
                if len(col_tags) > 0:
                    n_rows += 1
                    if len(col_tags) > n_cols:
                        n_cols = len(col_tags)

            # Create dataframe
            df_frame = pd.DataFrame(index=range(0, n_rows), columns=range(0, n_cols))

            # Create list to store rowspan values
            skip_index = [0 for i in range(0, n_cols)]

            # Start by iterating over each row in this table...
            row_counter = 0
            for row in tables_html[n].find_all("tr"):

                # Skip row if it's blank
                if len(row.find_all(["td", "th"])) == 0:
                    next

                else:

                    # Get all cells containing data in this row
                    columns = row.find_all(["td", "th"])
                    col_dim = []
                    row_dim = []
                    col_dim_counter = -1
                    row_dim_counter = -1
                    col_counter = -1
                    this_skip_index = copy.deepcopy(skip_index)

                    for col in columns:

                        # Determine cell dimensions
                        colspan = col.get("colspan")
                        if colspan is None:
                            col_dim.append(1)
                        else:
                            col_dim.append(int(colspan))
                        col_dim_counter += 1

                        rowspan = col.get("rowspan")
                        if rowspan is None:
                            row_dim.append(1)
                        else:
                            row_dim.append(int(rowspan))
                        row_dim_counter += 1

                        # Adjust column counter
                        if col_counter == -1:
                            col_counter = 0
                        else:
                            col_counter = col_counter + col_dim[col_dim_counter - 1]

                        while skip_index[col_counter] > 0:
                            col_counter += 1

                        # Get cell contents
                        cell_data = col.get_text()


                        # Insert data into cell
                        df_frame.iat[row_counter, col_counter] = cell_data
                        # Record column skipping index
                        if row_dim[row_dim_counter] > 1:
                            this_skip_index[col_counter] = row_dim[row_dim_counter]

                # Adjust row counter
                row_counter += 1

                # Adjust column skipping index
                skip_index = [i - 1 if i > 0 else i for i in this_skip_index]

            # Append dataframe to list of tables
            tables.append(df_frame)

        i = 0
        row_incr = 0
        #Setup the dataframe to be returned as the processed table
        new_df=tables[0]

        #Replace any empty values with the word "null"
        new_df = new_df.fillna("null")

        #Iterate through every row looking for null values
        for index, row in new_df.iterrows():
            row_incr += 1

            #Set the header incrementer
            i_incr = 0
            #Iterate through every column in the row
            while i < len(row):
                #If the value is null, let's act on it
                if row[i] == "null":
                    #Increment the header incrementer
                    i_incr += 1
                    #Set the value of the null header to the previous value plus the incrementer
                    row[i] = row[i-1]+str(i_incr)
                #Restart header incrementer
                i_incr=0
                #Increment the column counter
                i=i+1
            # Restart the column counter before next row
            i = 0
            if row_incr > 1:
                break
        new_df.reindex(new_df)

        new_df = new_df.drop(columns=[0]).drop([0])
        new_df.columns = new_df.iloc[headerrow]
        new_df.reindex(new_df)
        new_df = new_df.iloc[headerrow+1:]

        return new_df


    def partial_fit(self, df, options):
        # Incrementally fit a model

        pass

    def apply(self, df, options):
        # Apply a saved model
        # Modify df, a pandas DataFrame of the search results
        return df

    @staticmethod
    def register_codecs():
        # Add codecs to the codec manager
        pass
