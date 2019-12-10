import itertools
import pandas as pd
import numpy as np
import time

start = time.time()  # start runtime counter
df = pd.read_csv("Downloads/pokemon.csv")  # read in csv of fully-evolved Pokemon data
# list(df)  # list of df column names - useful to see what data has been collected
df = df[df["is_legendary"] == 0]  # remove legendary pokemon - many legendaries are allowed in competitive play
df = df[['abilities',  # trim df to contain only the columns we care about
        'against_bug',
        'against_dark',
        'against_dragon',
        'against_electric',
        'against_fairy',
        'against_fight',
        'against_fire',
        'against_flying',
        'against_ghost',
        'against_grass',
        'against_ground',
        'against_ice',
        'against_normal',
        'against_poison',
        'against_psychic',
        'against_rock',
        'against_steel',
        'against_water',
        'attack',
        'defense',
        'hp',
        'name',
        'sp_attack',
        'sp_defense',
        'speed',
        'type1',
        'type2']]
df["bst"] = df["hp"] + df["attack"] + df["defense"] + df["sp_attack"] + df["sp_defense"] + df["speed"]  # calculate BSTs
df['average_weakness'] = (df['against_bug'] # calculates a Pokemon's 'average weakness' to other types
                        + df['against_dark']
                        + df['against_dragon']
                        + df['against_electric']
                        + df['against_fairy']
                        + df['against_fight']
                        + df['against_fire']
                        + df['against_flying']
                        + df['against_ghost']
                        + df['against_grass']
                        + df['against_ground']
                        + df['against_ice']
                        + df['against_normal']
                        + df['against_poison']
                        + df['against_psychic']
                        + df['against_rock']
                        + df['against_steel']
                        + df['against_water']) / 18  
df['bst-weakness-ratio'] = df['bst'] / df['average_weakness']  # ratio of BST:avg weakness - the higher the better
df = df.loc[df["bst-weakness-ratio"] > np.mean(df["bst-weakness-ratio"])]  # remove all Pokemon w/ below-avg ratios
names = df["name"]  # pull out list of all names for creating combinations

combinations = itertools.combinations(names, 6) # create all possible combinations of 6 pokemon teams
top_10_teams = []  # list for storing top 10 teams
for x in combinations:
    ratio = sum(df.loc[df['name'].isin(x)]['bst-weakness-ratio'])  # pull out sum of team's ratio
    if(len(top_10_teams) != 10):
        top_10_teams.append((x, ratio))  # first 10 teams will automatically populate list
    else:
        top_10_teams.append((x, ratio))  # add team to list
        top_10_teams.sort(key=lambda x:x[1], reverse=True)  # sort list by descending ratios
        del top_10_teams[-1]  # drop team with the lowest ratio - only top 10 remain in list
elapsed = time.time() - start  # calculate total runtime 
top_10_teams  # print out the top 10 teams
