#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# ## get_subgenres

# In[3]:


def get_subgenres(genre_series = []):
    """
    Takes an input of a pd series of genres. 
    These are typically in CSV format, e.g. "Action,Adventure,Comedy"
    Parses these CSV strings and returns a list of the unique subgenres (e.g. "Action" separated from the others)
    """
    
    subgenres = []
    for g in genre_series:
        if type(g) == str:
            for subgenre in g.split(','):
                if not subgenre in subgenres:
                    subgenres.append(subgenre)
    return subgenres


# ## subgenre_counter

# In[4]:


def subgenre_counter(subgenres, df):
    """
    Takes in two positional arguments: 
        1. a list of subgenres (parsed by the parse_genres function)
        2. a pd DataFrame with a column labeled "genres"
    Counts how many times each subgenre occurs in the "genres" column of DataFrame.
    Results stored in genre_dict as key=subgenre, value=appearance_count
    """
    
    genre_dict = {}
    for long_genre in df.genres:
        for subgenre in subgenres:
            if subgenre in long_genre:
                if subgenre in genre_dict:
                    genre_dict[subgenre] += 1
                else:
                    genre_dict[subgenre] = 1
    return genre_dict



def prepare_data_for_roi_analysis():
    
    # import dataframes
    basics_df = pd.read_csv('data/imdb.title.basics.csv.gz')
    ratings_df = pd.read_csv('data/imdb.title.ratings.csv.gz')
    budgets_df = pd.read_csv('data/tn.movie_budgets.csv.gz')

    # 1st merge
    basics_ratings = pd.merge(basics_df, ratings_df, how='left', on='tconst') 

    # limit rows to ones that contain averagerating
    first_pct = str(1 - basics_ratings[basics_ratings.averagerating.isnull()].shape[0] / basics_ratings.shape[0])
    basics_ratings = basics_ratings[basics_ratings.averagerating.isnull() == False]

    # 2nd merge
    basics_ratings_budgets = pd.merge(basics_ratings, budgets_df, how='left', left_on='primary_title', right_on='movie')

    # limit rows to ones that contain production_budget
    second_pct = str(1 - basics_ratings_budgets[basics_ratings_budgets.production_budget.isnull()].shape[0] / basics_ratings_budgets.shape[0])
    basics_ratings_budgets = basics_ratings_budgets[basics_ratings_budgets.production_budget.isnull() == False]

    # print some info 
    print('Percent of rows left after first merge:', first_pct)
    print('Percent of rows left after second merge:', second_pct)
    
    return basics_ratings_budgets


def clean_data_prepare_features(df):
    
    # subset columns
    df = df[['primary_title','start_year','runtime_minutes','genres','averagerating','numvotes','production_budget','domestic_gross','worldwide_gross']].copy()
    
    # clean columns and convert to ints
    df.production_budget = df.production_budget.str.replace(',','').str.strip('$').astype(int)
    df.domestic_gross = df.domestic_gross.str.replace(',','').str.strip('$').astype(int)
    df.worldwide_gross = df.worldwide_gross.str.replace(',','').str.strip('$').apply(lambda x: int(x))
    
    # generate new features
    df['net_revenue'] = (df.worldwide_gross - df.production_budget) / 1000000
    df['log_numvotes'] = np.log(df.numvotes)
    df['scaled_rating'] = df.log_numvotes * df.averagerating / 127
    return df