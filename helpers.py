#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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



def plot_genre_comparison(performant_genres, top_subgenres_df):
    """
    plots two charts:
    1. the top performant genres as listed in the dataset
    2. the subgenres extracted from those top 10 performant genres
    """
    plt.figure(figsize=(20, 5))
    
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    g1 = sns.barplot(data=performant_genres, y=performant_genres.index, x='total_gross', orient='h', ax=ax1, color="#E67E17")
    g1.set_title("Top Performant Genres by Total Gross", fontsize=24, color="#E67E17")
    g1.set_xlabel("Total Gross (billions)", fontsize=20, color="#E67E17")
    g1.set_ylabel('Genres', fontsize=20, color="#E67E17")
        
    g2 = sns.barplot(data=top_subgenres_df, x="Count", y='Genre', ax=ax2, color="#366DA2")
    g2.set_title("Subgenre Counts in Top 10 Genres", fontsize=24, color="#366DA2")
    g2.set_xlabel("Count (in top 10)", fontsize=20,color="#366DA2")
    g2.set_ylabel("Subgenres", fontsize=20, color="#366DA2")
    
    plt.subplots_adjust(wspace=0.2)
    plt.show()


def plot_top_visually_enhanced_movies(top_visually_enhanced):
    """
    Shows the top 10 movies with some visually enhanced element (3D, IMAX) 
    ordered by total_gross revenue
    """
    sns.barplot(data=top_visually_enhanced, x='total_gross', y=top_visually_enhanced.index)
    plt.ylabel("Movie")
    plt.xlabel("Total Gross (billions)")
    plt.title("Top 10 Grossing Movies with Enhanced Visual Element")
    plt.show()


def plot_enhanced_attributes(enhanced_attributes):
    """
    plots total gross revenue of films grouped by their attribute enhancements 
    """
    sns.barplot(data=enhanced_attributes, x='total_gross', y=enhanced_attributes.index)
    plt.xticks(rotation=0.5)
    plt.title("Total Gross of Enhanced Films", fontsize=24, color="#366DA2")
    plt.xlabel("Total Gross (billions)", fontsize=20, color="#366DA2")
    plt.ylabel("Enhancement", fontsize=20, color="#366DA2")
    plt.show()

def plot_3d_trend(three_d):
    """
    Plots the focus on 3D films over an 8 year period from 2010-2018 
    """
    sns.lineplot(data=three_d.reset_index(), x='year', y='gross_by_num_films')
    plt.title("Average gross per 3D Film per Year", fontsize=20, color="#366DA2")
    plt.xlabel("Year", fontsize=20, color="#366DA2")
    plt.ylabel("Average Gross Revenue (billions)", fontsize=15, color="#366DA2")
    plt.show()

def plot_visually_enhanced_revenue_distribution(split_3d, imax, final_df):
    """
    plots the distribution of gross revenue for 3D, IMAX, and Standard definition films.
    """

    f = plt.figure(figsize=(18, 6))

    sns.distplot(a=split_3d.groupby('primary_title').sum().total_gross, bins=50, hist=True, hist_kws=dict(alpha=1), kde_kws=dict(alpha=0))
    sns.distplot(a=imax.total_gross, bins=50, kde=False, hist_kws=dict(alpha=0.5), kde_kws=dict(alpha=0))
    sns.distplot(a=final_df.total_gross, bins=50, hist_kws=dict(alpha=0.25), kde_kws=dict(alpha=0))
    
    plt.legend(['3D', 'IMAX', "Standard"], fontsize=30)

    plt.xlabel("Total Gross (billions)", fontsize=30)
    plt.ylabel('Count', fontsize=30)
    plt.title("Distribution of Gross Revenue by Visual Enhancement", fontsize=30)
    plt.xlim([0, 1.5])
    plt.show()