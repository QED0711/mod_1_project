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

