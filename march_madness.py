from __future__ import annotations

import pandas as pd
import numpy as np
import itertools
import collections
from sklearn.model_selection import cross_val_score, GridSearchCV
import logging
from sklearn.model_selection import train_test_split, cross_validate
from typing import List, Tuple, Dict
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import tree
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
from sklearn.ensemble import GradientBoostingRegressor
import math
import csv
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import urllib
from sklearn.svm import LinearSVC
from utils import *

logging.basicConfig(level=logging.DEBUG)


training_mean = 0
training_std = 0
training_min = 0
training_max = 0
#TODO add logging
#TODO add error catching
#TODO add files to github
def add_poss_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Adds possession statistics to dataframe
    
    This calculates number of possessions and adds columns to indicate possessions and per possession stats 

    Args:
        df (DataFrame): A DataFrame containing detailed NCAA game statistics
    Returns:
        DataFrame
    """

    # handle possessions for detailed tourney results

    #adds possession column
    df['Wposs'] = df['WFGA'] - df['WOR'] + df[
        'WTO'] + .44 * df['WFTA']
    df['Lposs'] = df['LFGA'] - df['LOR'] + df[
        'LTO'] + .44 * df['LFTA']

    #determines 1st and 3rd quantiles for possession numbers
    tourney_poss_quantiles = df['Wposs'].quantile([.25, .5, .75])
    tourney_poss_low = tourney_poss_quantiles.iloc[0]
    tourney_poss_mean = tourney_poss_quantiles.iloc[1]
    tourney_poss_high = tourney_poss_quantiles.iloc[2]

    #adds a regularized column for possessions
    df['Wposs_sc'] = (df['Wposs'] - tourney_poss_low) / (
                tourney_poss_high - tourney_poss_low)
    df['Lposs_sc'] = (df['Lposs'] - tourney_poss_low) / (
                tourney_poss_high - tourney_poss_low)

    #adds columns for stats per possession
    df['WfgmPerPoss'] = df['WFGM'] / df['Wposs']
    df['WfgaPerPoss'] = df['WFGA'] / df['Wposs']
    df['Wfgm3PerPoss'] = df['WFGM3'] / df['Wposs']
    df['Wfga3PerPoss'] = df['WFGA3'] / df['Wposs']
    df['WftmPerPoss'] = df['WFTM'] / df['Wposs']
    df['WftaPerPoss'] = df['WFTA'] / df['Wposs']
    df['WorPerPoss'] = df['WOR'] / df['Wposs']
    df['WdrPerPoss'] = df['WDR'] / df['Wposs']
    df['WastPerPoss'] = df['WAst'] / df['Wposs']
    df['WtoPerPoss'] = df['WTO'] / df['Wposs']
    df['WstlPerPoss'] = df['WStl'] / df['Wposs']
    df['WblkPerPoss'] = df['WBlk'] / df['Wposs']
    df['WpfPerPoss'] = df['WPF'] / df['Wposs']

    df['LfgmPerPoss'] = df['LFGM'] / df['Lposs']
    df['LfgaPerPoss'] = df['LFGA'] / df['Lposs']
    df['Lfgm3PerPoss'] = df['LFGM3'] / df['Lposs']
    df['Lfga3PerPoss'] = df['LFGA3'] / df['Lposs']
    df['LftmPerPoss'] = df['LFTM'] / df['Lposs']
    df['LftaPerPoss'] = df['LFTA'] / df['Lposs']
    df['LorPerPoss'] = df['LOR'] / df['Lposs']
    df['LdrPerPoss'] = df['LDR'] / df['Lposs']
    df['LastPerPoss'] = df['LAst'] / df['Lposs']
    df['LtoPerPoss'] = df['LTO'] / df['Lposs']
    df['LstlPerPoss'] = df['LStl'] / df['Lposs']
    df['LblkPerPoss'] = df['LBlk'] / df['Lposs']
    df['LpfPerPoss'] = df['LPF'] / df['Lposs']
    return df

def checkPower6Conference(team_id: int) -> int:
    """Checks if a team is in a power conference

    This checks if a team is in oneof the top 6 larger conferenes in the NCAA it returns 1 if so and 0 if not

    Args:
        team_id (int): The integer id of a given team
    Returns:
        int
    """

    teamName = teams_pd.values[team_id-1101][1]
    if (teamName in listACCteams or teamName in listBig10teams or teamName in listBig12teams
       or teamName in listSECteams or teamName in listPac12teams or teamName in listBigEastteams):
        return 1
    else:
        return 0

def getTeamID(name: str) -> int:
    """Gets team id from name of school

    Args:
        name (str): The name of an NCAA school
    Returns:
        int
    """
    return teams_pd[teams_pd['Team_Name'] == name].values[0][0]


def getTeamName(team_id: int) -> str:
    """Gets team name from team id

    Args:
        team_id (int): Integer id of an NCAA team
    Returns:
        str
    """

    return teams_pd[teams_pd['Team_Id'] == team_id].values[0][1]


def getNumChampionships(team_id: int) -> int:
    """Gets the number of chapionships won by a team based on team id

    Args:
        team_id (int): Integer id of an NCAA team
    Returns:
        int
    """

    name = getTeamName(team_id)
    return NCAAChampionsList.count(name)

# TODO Check and see if this is used or not
# def getListForURL(team_list):
#     """Gets the number of championships won by a team based on team id
#
#     Args:
#         team_id (int): Integer id of an NCAA team
#     Returns:
#         int
#     """
#     team_list = [x.lower() for x in team_list]
#     team_list = [t.replace(' ', '-') for t in team_list]
#     team_list = [t.replace('st', 'state') for t in team_list]
#     team_list = [t.replace('northern-dakota', 'north-dakota') for t in team_list]
#     team_list = [t.replace('nc-', 'north-carolina-') for t in team_list]
#     team_list = [t.replace('fl-', 'florida-') for t in team_list]
#     team_list = [t.replace('ga-', 'georgia-') for t in team_list]
#     team_list = [t.replace('lsu', 'louisiana-state') for t in team_list]
#     team_list = [t.replace('maristate', 'marist') for t in team_list]
#     team_list = [t.replace('stateate', 'state') for t in team_list]
#     team_list = [t.replace('northernorthern', 'northern') for t in team_list]
#     team_list = [t.replace('usc', 'southern-california') for t in team_list]
#     base = 'http://www.sports-reference.com/cbb/schools/'
#     for team in team_list:
#         url = base + team + '/'

# Function for handling the annoying cases of Florida and FL, as well as State and St
def handleCases(arr: list[str]) -> list[str]:
    """Handles school names with Florida or State

    Some conventions spell out Florida and some use FL. Some spell out State and some use St. This standardizes these
    naming conventions in a given list of school names.

    Args:
        arr (List[str]): List of school names
    Returns:
        List[str]
    """

    indices = []
    listLen = len(arr)
    for i in range(listLen):
        if (arr[i] == 'St' or arr[i] == 'FL'):
            indices.append(i)
    for p in indices:
        arr[p - 1] = arr[p - 1] + ' ' + arr[p]
    for i in range(len(indices)):
        arr.remove(arr[indices[i] - i])
    return arr



def checkConferenceChamp(team_id: int, year: int) -> int:
    """Checks if a team won their conference in a given year.

    If the team won their conference, the function returns 1. Otherwise, it returns 0.

    Args:
        team_id (int): Integer id of an NCAA team
        year (int): Full year value
    Returns:
        int
    """

    year_conf_pd = conference_pd[conference_pd['Year'] == year]
    champs = year_conf_pd['Regular Season Champ'].tolist()
    # For handling cases where there is more than one champion
    champs_separated = [words for segments in champs for words in segments.split()]
    name = getTeamName(team_id)
    champs_separated = handleCases(champs_separated)
    if name in champs_separated:
        return 1
    else:
        return 0



def checkConferenceTourneyChamp(team_id: int, year: int) -> int:
    """Checks if a team won their conference championship game in a given year.

    If the team won their conference championship, the function returns 1. Otherwise it returns 0.

    Args:
        team_id (int): Integer id of an NCAA team
        year (int): Full year value
    Returns:
        int
    """

    year_conf_pd = conference_pd[conference_pd['Year'] == year]
    champs = year_conf_pd['Tournament Champ'].tolist()
    name = getTeamName(team_id)
    if name in champs:
        return 1
    else:
        return 0



def getTourneyAppearances(team_id: int) -> int:
    """Checks how many times a team has played in the NCAA tournamnet.

    Args:
        team_id (int): Integer id of an NCAA team
    Returns:
        int
    """
    return len(tourney_seeds_pd[tourney_seeds_pd['TeamID'] == team_id].index)



def handleDifferentCSV(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans school naming differences between different data sources

    The stats CSV is a little different in terms of naming so this is just some data cleaning
    Args:
        df (DataFrame): DataFrame including names of schoolsm
    Returns:
        DataFrame
    """

    df['School'] = df['School'].replace('(State)', 'St', regex=True)
    df['School'] = df['School'].replace('Albany (NY)', 'Albany NY')
    df['School'] = df['School'].replace('Boston University', 'Boston Univ')
    df['School'] = df['School'].replace('Central Michigan', 'C Michigan')
    df['School'] = df['School'].replace('(Eastern)', 'E', regex=True)
    df['School'] = df['School'].replace('Louisiana St', 'LSU')
    df['School'] = df['School'].replace('North Carolina St', 'NC State')
    df['School'] = df['School'].replace('Southern California', 'USC')
    df['School'] = df['School'].replace('University of California', 'California', regex=True)
    df['School'] = df['School'].replace('American', 'American Univ')
    df['School'] = df['School'].replace('Arkansas-Little Rock', 'Ark Little Rock')
    df['School'] = df['School'].replace('Arkansas-Pine Bluff', 'Ark Pine Bluff')
    df['School'] = df['School'].replace('Bowling Green St', 'Bowling Green')
    df['School'] = df['School'].replace('Brigham Young', 'BYU')
    df['School'] = df['School'].replace('Cal Poly', 'Cal Poly SLO')
    df['School'] = df['School'].replace('Centenary (LA)', 'Centenary')
    df['School'] = df['School'].replace('Central Connecticut St', 'Central Conn')
    df['School'] = df['School'].replace('Charleston Southern', 'Charleston So')
    df['School'] = df['School'].replace('Coastal Carolina', 'Coastal Car')
    df['School'] = df['School'].replace('College of Charleston', 'Col Charleston')
    df['School'] = df['School'].replace('Cal St Fullerton', 'CS Fullerton')
    df['School'] = df['School'].replace('Cal St Sacramento', 'CS Sacramento')
    df['School'] = df['School'].replace('Cal St Bakersfield', 'CS Bakersfield')
    df['School'] = df['School'].replace('Cal St Northridge', 'CS Northridge')
    df['School'] = df['School'].replace('East Tennessee St', 'ETSU')
    df['School'] = df['School'].replace('Detroit Mercy', 'Detroit')
    df['School'] = df['School'].replace('Fairleigh Dickinson', 'F Dickinson')
    df['School'] = df['School'].replace('Florida Atlantic', 'FL Atlantic')
    df['School'] = df['School'].replace('Florida Gulf Coast', 'FL Gulf Coast')
    df['School'] = df['School'].replace('Florida International', 'Florida Intl')
    df['School'] = df['School'].replace('George Washington', 'G Washington')
    df['School'] = df['School'].replace('Georgia Southern', 'Ga Southern')
    df['School'] = df['School'].replace('Gardner-Webb', 'Gardner Webb')
    df['School'] = df['School'].replace('Illinois-Chicago', 'IL Chicago')
    df['School'] = df['School'].replace('Kent St', 'Kent')
    df['School'] = df['School'].replace('Long Island University', 'Long Island')
    df['School'] = df['School'].replace('Loyola Marymount', 'Loy Marymount')
    df['School'] = df['School'].replace('Loyola (MD)', 'Loyola MD')
    df['School'] = df['School'].replace('Loyola (IL)', 'Loyola-Chicago')
    df['School'] = df['School'].replace('Massachusetts', 'MA Lowell')
    df['School'] = df['School'].replace('Maryland-Eastern Shore', 'MD E Shore')
    df['School'] = df['School'].replace('Miami (FL)', 'Miami FL')
    df['School'] = df['School'].replace('Miami (OH)', 'Miami OH')
    df['School'] = df['School'].replace('Missouri-Kansas City', 'Missouri KC')
    df['School'] = df['School'].replace('Monmouth', 'Monmouth NJ')
    df['School'] = df['School'].replace('Mississippi Valley St', 'MS Valley St')
    df['School'] = df['School'].replace('Montana State', 'Montana St')
    df['School'] = df['School'].replace('Northern Colorado', 'N Colorado')
    df['School'] = df['School'].replace('North Dakota St', 'N Dakota St')
    df['School'] = df['School'].replace('Northern Illinois', 'N Illinois')
    df['School'] = df['School'].replace('Northern Kentucky', 'N Kentucky')
    df['School'] = df['School'].replace('North Carolina A&T', 'NC A&T')
    df['School'] = df['School'].replace('North Carolina Central', 'NC Central')
    df['School'] = df['School'].replace('Pennsylvania', 'Penn')
    df['School'] = df['School'].replace('South Carolina St', 'S Carolina St')
    df['School'] = df['School'].replace('Southern Illinois', 'S Illinois')
    df['School'] = df['School'].replace('UC-Santa Barbara', 'Santa Barbara')
    df['School'] = df['School'].replace('Southeastern Louisiana', 'SE Louisiana')
    df['School'] = df['School'].replace('Southeast Missouri St', 'SE Missouri St')
    df['School'] = df['School'].replace('Stephen F. Austin', 'SF Austin')
    df['School'] = df['School'].replace('Southern Methodist', 'SMU')
    df['School'] = df['School'].replace('Southern Mississippi', 'Southern Miss')
    df['School'] = df['School'].replace('Southern', 'Southern Univ')
    df['School'] = df['School'].replace('St. Bonaventure', 'St Bonaventure')
    df['School'] = df['School'].replace('St. Francis (NY)', 'St Francis NY')
    df['School'] = df['School'].replace('Saint Francis (PA)', 'St Francis PA')
    df['School'] = df['School'].replace('St. John\'s (NY)', 'St John\'s')
    df['School'] = df['School'].replace('Saint Joseph\'s', 'St Joseph\'s PA')
    df['School'] = df['School'].replace('Saint Louis', 'St Louis')
    df['School'] = df['School'].replace('Saint Mary\'s (CA)', 'St Mary\'s CA')
    df['School'] = df['School'].replace('Mount Saint Mary\'s', 'Mt St Mary\'s')
    df['School'] = df['School'].replace('Saint Peter\'s', 'St Peter\'s')
    df['School'] = df['School'].replace('Texas A&M-Corpus Christian', 'TAM C. Christian')
    df['School'] = df['School'].replace('Texas Christian', 'TCU')
    df['School'] = df['School'].replace('Tennessee-Martin', 'TN Martin')
    df['School'] = df['School'].replace('Texas-Rio Grande Valley', 'UTRGV')
    df['School'] = df['School'].replace('Texas Southern', 'TX Southern')
    df['School'] = df['School'].replace('Alabama-Birmingham', 'UAB')
    df['School'] = df['School'].replace('UC-Davis', 'UC Davis')
    df['School'] = df['School'].replace('UC-Irvine', 'UC Irvine')
    df['School'] = df['School'].replace('UC-Riverside', 'UC Riverside')
    df['School'] = df['School'].replace('Central Florida', 'UCF')
    df['School'] = df['School'].replace('Louisiana-Lafayette', 'ULL')
    df['School'] = df['School'].replace('Louisiana-Monroe', 'ULM')
    df['School'] = df['School'].replace('Maryland-Baltimore County', 'UMBC')
    df['School'] = df['School'].replace('North Carolina-Asheville', 'UNC Asheville')
    df['School'] = df['School'].replace('North Carolina-Greensboro', 'UNC Greensboro')
    df['School'] = df['School'].replace('North Carolina-Wilmington', 'UNC Wilmington')
    df['School'] = df['School'].replace('Nevada-Las Vegas', 'UNLV')
    df['School'] = df['School'].replace('Texas-Arlington', 'UT Arlington')
    df['School'] = df['School'].replace('Texas-San Antonio', 'UT San Antonio')
    df['School'] = df['School'].replace('Texas-El Paso', 'UTEP')
    df['School'] = df['School'].replace('Virginia Commonwealth', 'VA Commonwealth')
    df['School'] = df['School'].replace('Western Carolina', 'W Carolina')
    df['School'] = df['School'].replace('Western Illinois', 'W Illinois')
    df['School'] = df['School'].replace('Western Kentucky', 'WKU')
    df['School'] = df['School'].replace('Western Michigan', 'W Michigan')
    df['School'] = df['School'].replace('Abilene Christian', 'Abilene Chr')
    df['School'] = df['School'].replace('Montana State', 'Montana St')
    df['School'] = df['School'].replace('Central Arkansas', 'Cent Arkansas')
    df['School'] = df['School'].replace('Houston Baptist', 'Houston Bap')
    df['School'] = df['School'].replace('South Dakota St', 'S Dakota St')
    df['School'] = df['School'].replace('Maryland-Eastern Shore', 'MD E Shore')
    return df


def season_totals(wstat: str, lstat: str, gamesWon: pd.DataFrame, df: pd.DataFrame, team_id: int) -> Tuple[int,int]:
    """Adds the total number of a statistic for a team

    For a given DataFrame, which is often already filtered by year, this adds up the total of the provided statistic. It
    also returns the total number of games the team played in the given filtered DataFrame.

    Args:
        wstat (str): the name of a statistic from the winning team represented in the DataFrame
        lstat (str): the name of a statistic from the losing team represented in the DataFrame
        gamesWon (DataFrame): DataFrame with all the games which the given team has won
        df (DataFrame): DataFrame of game statistics
        team_id (int): the integer id of an NCAA team
    Returns:
        (int, int)
    """

    total = gamesWon[wstat].sum()
    gamesLost = df[df.LTeamID == team_id]
    totalGames = gamesWon.append(gamesLost)
    numGames = len(totalGames.index)
    total += gamesLost[lstat].sum()

    return total, numGames

def getSeasonData(team_id: int, year: int) -> List[float]:
    """Gets the full season data for a given team in a given year

    This function looks at a team in a given year and returns a list of statistics which becomes the team's vector for
    that season. This list of statistics can be edited and there are multiple iterations commented out in the code which
    have been used at various stages of testing.

    Args:
        team_id (int): Integer id of an NCAA team
        year (int): Full year value
    Returns:
        List[float]
    """

    # The data frame below holds stats for every single game in the given year
    year_data_pd = reg_season_detailed_pd[reg_season_detailed_pd['Season'] == year]
    # Finding number of points per game
    gamesWon = year_data_pd[year_data_pd.WTeamID == team_id]
    gamesLost = year_data_pd[year_data_pd.LTeamID == team_id]


    totalPointsScored = season_totals('WScore','LScore',gamesWon,year_data_pd,team_id)[0]
    numGames = season_totals('WScore','LScore',gamesWon,year_data_pd,team_id)[1]

    # Finding number of possessions in season
    totalPoss = season_totals('Wposs', 'Lposs', gamesWon, year_data_pd, team_id)[0]

    # Finding number of fgm in season
    totalFgm = season_totals('WFGM', 'LFGM', gamesWon, year_data_pd, team_id)[0]

    # Finding number of fga in season
    totalFga = season_totals('WFGA', 'LFGA', gamesWon, year_data_pd, team_id)[0]

    # Finding number of fgm3 in season
    totalFgm3 = season_totals('WFGM3', 'LFGM3', gamesWon, year_data_pd, team_id)[0]

    # Finding number of fga3 in season
    totalFga3 = season_totals('WFGA3', 'LFGA3', gamesWon, year_data_pd, team_id)[0]

    # Finding number of ftm in season
    totalFtm = season_totals('WFTM', 'LFTM', gamesWon, year_data_pd, team_id)[0]

    # Finding number of fta in season
    totalFta = season_totals('WFTA', 'LFTA', gamesWon, year_data_pd, team_id)[0]

    # Finding number of or in season
    totalOr = season_totals('WOR', 'LOR', gamesWon, year_data_pd, team_id)[0]

    # Finding number of dr in season
    totalDr = season_totals('WDR', 'LDR', gamesWon, year_data_pd, team_id)[0]

    totalReb = totalOr + totalDr

    # Finding number of blk in season
    totalBlk = season_totals('WBlk', 'LBlk', gamesWon, year_data_pd, team_id)[0]

    # Finding number of pf in season
    totalPf = season_totals('WPF', 'LPF', gamesWon, year_data_pd, team_id)[0]

    # Finding number of to in season
    totalTo = season_totals('WTO', 'LTO', gamesWon, year_data_pd, team_id)[0]

    # Finding number of ast in season
    totalAst = season_totals('WAst', 'LAst', gamesWon, year_data_pd, team_id)[0]

    # Finding number of Stl in season
    totalStl = season_totals('WStl', 'LStl', gamesWon, year_data_pd, team_id)[0]

    # Finding number of points per game allowed
    totalPointsAllowed = gamesWon['LScore'].sum()
    totalPointsAllowed += gamesLost['WScore'].sum()

    #    stats_SOS_pd = pd.read_csv('C:/Users/Lenovo/Documents/Data/MMStats/MMStats_'+str(year)+'.csv')
    #    stats_SOS_pd = handleDifferentCSV(stats_SOS_pd)
    #    ratings_pd = pd.read_csv('C:/Users/Lenovo/Documents/Data/RatingStats/RatingStats_'+str(year)+'.csv')
    #    ratings_pd = handleDifferentCSV(ratings_pd)
    #
    #    name = getTeamName(team_id)
    #    team = stats_SOS_pd[stats_SOS_pd['School'] == name]
    #    team_rating = ratings_pd[ratings_pd['School'] == name]
    #    if (len(team.index) == 0 or len(team_rating.index) == 0): #Can't find the team
    #        total3sMade = 0
    #        totalTurnovers = 0
    #        totalAssists = 0
    #        sos = 0
    #        totalRebounds = 0
    #        srs = 0
    #        totalSteals = 0
    #    else:
    #        total3sMade = team['X3P'].values[0]
    #        totalTurnovers = team['TOV'].values[0]
    #        if (math.isnan(totalTurnovers)):
    #            totalTurnovers = 0
    #        totalAssists = team['AST'].values[0]
    #        if (math.isnan(totalAssists)):
    #            totalAssists = 0
    #        sos = team['SOS'].values[0]
    #        srs = team['SRS'].values[0]
    #        totalRebounds = team['TRB'].values[0]
    #        if (math.isnan(totalRebounds)):
    #            totalRebounds = 0
    #        totalSteals = team['STL'].values[0]
    #        if (math.isnan(totalSteals)):
    #            totalSteals = 0

    # Finding tournament seed for that year
    tourneyYear = tourney_seeds_pd[tourney_seeds_pd['Season'] == year]
    seed = tourneyYear[tourneyYear['TeamID'] == team_id]
    if len(seed.index) != 0:
        seed = seed.values[0][1]
        tournamentSeed = int(seed[1:3])
    else:
        tournamentSeed = 25  # Not sure how to represent if a team didn't make the tourney

    # Finding number of wins and losses
    numWins = len(gamesWon.index)
    # There are some teams who may have dropped to Division 2, so they won't have games
    # a certain year. In this case, we don't want to divide by 0, so we'll just set the
    # averages to 0 instead
    if numGames == 0:
        avgPointsScored = 0
        avgPointsAllowed = 0
        avg3sMade = 0
        avgTurnovers = 0
        avgAssists = 0
        avgRebounds = 0
        avgSteals = 0
        totalfgmPerPoss = 0
        totalfgaPerPoss = 0
        totalfgm3PerPoss = 0
        totalfga3PerPoss = 0
        totalftmPerPoss = 0
        totalftaPerPoss = 0
        totalorPerPoss = 0
        totaldrPerPoss = 0
        totalastPerPoss = 0
        totaltoPerPoss = 0
        totalstlPerPoss = 0
        totalblkPerPoss = 0
        totalpfPerPoss = 0
    else:
        avgPointsScored = totalPointsScored / numGames
        avgPointsAllowed = totalPointsAllowed / numGames
        avg3sMade = totalFgm3 / numGames
        avgTurnovers = totalTo / numGames
        avgAssists = totalAst / numGames
        avgRebounds = totalReb / numGames
        avgSteals = totalStl / numGames
        totalfgmPerPoss = totalFgm / totalPoss
        totalfgaPerPoss = totalFga / totalPoss
        totalfgm3PerPoss = totalFgm3 / totalPoss
        totalfga3PerPoss = totalFga3 / totalPoss
        totalftmPerPoss = totalFtm / totalPoss
        totalftaPerPoss = totalFta / totalPoss
        totalorPerPoss = totalOr / totalPoss
        totaldrPerPoss = totalDr / totalPoss
        totalastPerPoss = totalAst / totalPoss
        totaltoPerPoss = totalTo / totalPoss
        totalstlPerPoss = totalStl / totalPoss
        totalblkPerPoss = totalBlk / totalPoss
        totalpfPerPoss = totalPf / totalPoss
    # return [numWins, sos, srs]
    # return [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id), avg3sMade, avgTurnovers,
    #        tournamentSeed, getStrengthOfSchedule(team_id, year), getTourneyAppearances(team_id)]
    # This is the full feature set based on intuition
    # features = [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id), avg3sMade, avgAssists,
    #         avgTurnovers,
    #         checkConferenceChamp(team_id, year), checkConferenceTourneyChamp(team_id, year), tournamentSeed,
    #         avgRebounds, avgSteals, getTourneyAppearances(team_id), getNumChampionships(team_id), totalPoss,
    #         totalfgmPerPoss, totalfgaPerPoss, totalfgm3PerPoss, totalfga3PerPoss, totalftmPerPoss, totalftaPerPoss,
    #         totalorPerPoss, totaldrPerPoss, totalastPerPoss, totaltoPerPoss, totalstlPerPoss, totalblkPerPoss,
    #         totalpfPerPoss]
    # This is after RFE on training data normalized with the mean
    features = [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id), avgAssists, avgTurnovers,
                tournamentSeed, getTourneyAppearances(team_id), totalfgmPerPoss, totalftmPerPoss, totalftaPerPoss,
                totaldrPerPoss, totalastPerPoss]
    # features = [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id),
    #             avgAssists, avgTurnovers, tournamentSeed, getTourneyAppearances(team_id),
    #             totalPoss, totalfgmPerPoss, totalftmPerPoss, totaldrPerPoss, totalastPerPoss]
    float_features = [float(x) for x in features]
    return float_features

    # return [sos,srs,totalPoss, totalfgmPerPoss, totalfgaPerPoss, totalfgm3PerPoss, totalfga3PerPoss, totalftmPerPoss, totalftaPerPoss,
    #       totalorPerPoss, totaldrPerPoss, totalastPerPoss, totaltoPerPoss, totalstlPerPoss, totalblkPerPoss, totalpfPerPoss]


#TODO this will need to be for 2023
def get2022Data(team_id: int) -> List[float]:
    """Gets the full season data for a given team in 2022

    This function looks at a team in 2022 and returns a list of statistics which becomes the team's vector for
    that season. This list of statistics can be edited and there are multiple iterations commented out in the code which
    have been used at various stages of testing. The list of features needs to match the list from getSeasonData.

    Args:
        team_id (int): Integer id of an NCAA team
    Returns:
        List[float]
    """

    # Finding number of points per game
    team_data = this_year_pd[this_year_pd.iloc[:, 0] == team_id]
    numWins = team_data.iloc[:, 3]
    avgPointsScored = team_data.iloc[:, 14]
    avgPointsAllowed = team_data.iloc[:, 15]
    avg3sMade = team_data.iloc[:, 20]
    avgAssists = team_data.iloc[:, 28]
    avgTurnovers = team_data.iloc[:, 31]
    tournamentSeed = team_data.iloc[:, 33]
    sos = team_data.iloc[:, 7]
    srs = team_data.iloc[:, 6]
    avgRebounds = team_data.iloc[:, 27]
    avgSteals = team_data.iloc[:, 29]
    reg_conf_champ = team_data.iloc[:, 34]
    conf_tourn_champ = team_data.iloc[:, 35]
    totalPoss = team_data.iloc[:, 36]
    totalfgmPerPoss = team_data.iloc[:, 38]
    totalfgaPerPoss = team_data.iloc[:, 39]
    totalfgm3PerPoss = team_data.iloc[:, 40]
    totalfga3PerPoss = team_data.iloc[:, 41]
    totalftmPerPoss = team_data.iloc[:, 42]
    totalftaPerPoss = team_data.iloc[:, 43]
    totalorPerPoss = team_data.iloc[:, 44]
    totaldrPerPoss = team_data.iloc[:, 45]
    totalastPerPoss = team_data.iloc[:, 46]
    totaltoPerPoss = team_data.iloc[:, 47]
    totalstlPerPoss = team_data.iloc[:, 48]
    totalblkPerPoss = team_data.iloc[:, 49]
    totalpfPerPoss = team_data.iloc[:, 50]
    #the vector needs to match the vector in getSeasonData
    # features = [float(numWins.iloc[0]), float(avgPointsScored), float(avgPointsAllowed), float(checkPower6Conference(team_id)),
    #         float(avg3sMade), float(avgAssists), float(avgTurnovers),
    #         float(reg_conf_champ), float(conf_tourn_champ), float(tournamentSeed),
    #         float(avgRebounds), float(avgSteals), float(getTourneyAppearances(team_id)),
    #         float(getNumChampionships(team_id)), totalPoss.iloc[0],
    #         totalfgmPerPoss.iloc[0], totalfgaPerPoss.iloc[0], totalfgm3PerPoss.iloc[0], totalfga3PerPoss.iloc[0],
    #         totalftmPerPoss.iloc[0], totalftaPerPoss.iloc[0],
    #         totalorPerPoss.iloc[0], totaldrPerPoss.iloc[0], totalastPerPoss.iloc[0], totaltoPerPoss.iloc[0],
    #         totalstlPerPoss.iloc[0], totalblkPerPoss.iloc[0], totalpfPerPoss.iloc[0]]

    features = [float(numWins.iloc[0]), float(avgPointsScored), float(avgPointsAllowed), float(checkPower6Conference(team_id)),
            float(avgAssists), float(avgTurnovers), float(tournamentSeed), float(getTourneyAppearances(team_id)),
            totalfgmPerPoss.iloc[0], totalftmPerPoss.iloc[0], totalftaPerPoss.iloc[0], totaldrPerPoss.iloc[0], totalastPerPoss.iloc[0]]

    return features


def compareTwoTeams(id_1: int, id_2: int, year: int) -> List[float]:
    """Compares two teams in a given year and gives a combined vector for the matchup

    This function looks at two teams in a given year. It subtracts one team's vector from the other to create a matchup
    vector.

    Args:
        id_1 (int): Integer id of the first NCAA team
        id_2 (int): Integer id of the second NCAA team
        year (int): Full year
    Returns:
        List[float]
    """

    if year==2022:
        team_1 = get2022Data(id_1)
        team_2 = get2022Data(id_2)
    else:
        team_1 = getSeasonData(id_1, year)
        team_2 = getSeasonData(id_2, year)
    diff = [a - b for a, b in zip(team_1, team_2)]
    return diff


def createSeasonDict(year: int) -> Dict[int,List[float]]:
    """Creates a dictionary of team vectors for all teams in a given year.

    This function looks at all teams in a given year and calculates their team vector. It then creates a dictionary
    where the team_id is the key and the vector is the value.

    Args:
        year (int): Full year
    Returns:
        Dict[int,List[float]]
    """

    seasonDictionary = collections.defaultdict(list)
    for team in teamList:
        team_id = teams_pd[teams_pd['Team_Name'] == team].values[0][0]
        team_vector = getSeasonData(team_id, year)
        seasonDictionary[team_id] = team_vector
    return seasonDictionary



def getHomeStat(row: pd.Series) -> int:
    """Calculates if a team was at home, away, or at a neutral site.

    This function looks at a matchup and if the team was the home team returns 1, if away it returns -1. If neutral it
    returns 0.

    Args:
        row (pd.Series): A row of a dataframe representing a matchup
    Returns:
        int
    """
    home = 0
    if row == 'H':
        home = 1
    elif row == 'A':
        home = -1
    elif row == 'N':
        home = 0
    return home



def createTrainingSet(years: range) -> Tuple[np.array,np.array]:
    """Creates a training set and a training target set

    This function iterates through a range of years and creates a training set of matchup vectors as well as a target
    vector indicating if the first named team in the matchup won or lost.

    Args:
        years (range): A range object of years to include in the training set
    Returns:
        Tuple[np.array,np.array]
    """

    totalNumGames = 0
    #loop through years
    for year in years:
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
        totalNumGames += len(season.index)
        tourney = tourney_compact_pd[tourney_compact_pd['Season'] == year]
        totalNumGames += len(tourney.index)
    numFeatures = len(getSeasonData(1181,2012)) #Just choosing a random team and seeing the dimensionality of the vector
    x_Train = np.zeros(( totalNumGames, numFeatures + 1))
    y_Train = np.zeros( totalNumGames )
    indexCounter = 0
    for year in years:
        team_vectors = createSeasonDict(year)
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
        numGamesInSeason = len(season.index)
        tourney = tourney_compact_pd[tourney_compact_pd['Season'] == year]
        numGamesInSeason += len(tourney.index)
        xTrainSeason = np.zeros(( numGamesInSeason, numFeatures + 1))
        yTrainSeason = np.zeros(( numGamesInSeason ))
        counter = 0
        for index, row in season.iterrows():
            w_team = row['WTeamID']
            w_vector = team_vectors[w_team]
            l_team = row['LTeamID']
            l_vector = team_vectors[l_team]
            diff = [a - b for a, b in zip(w_vector, l_vector)]
            home = getHomeStat(row['WLoc'])
            if (counter % 2 == 0):
                diff.append(home)
                xTrainSeason[counter] = diff
                yTrainSeason[counter] = 1
            else:
                diff.append(-home)
                xTrainSeason[counter] = [ -p for p in diff]
                yTrainSeason[counter] = 0
            counter += 1
        for index, row in tourney.iterrows():
            w_team = row['WTeamID']
            w_vector = team_vectors[w_team]
            l_team = row['LTeamID']
            l_vector = team_vectors[l_team]
            diff = [a - b for a, b in zip(w_vector, l_vector)]
            home = 0 #All tournament games are neutral
            if (counter % 2 == 0):
                diff.append(home)
                xTrainSeason[counter] = diff
                yTrainSeason[counter] = 1
            else:
                diff.append(-home)
                xTrainSeason[counter] = [ -p for p in diff]
                yTrainSeason[counter] = 0
            counter += 1
        x_Train[indexCounter:numGamesInSeason+indexCounter] = xTrainSeason
        y_Train[indexCounter:numGamesInSeason+indexCounter] = yTrainSeason
        indexCounter += numGamesInSeason
    return x_Train, y_Train


#TODO normalize inputs
def normalizeInput(arr: List[float]) -> List[float]:
    """Normalizes values in a vector

    This function scales the values in the vectors between the max and the min values in the data.

    Args:
        arr (List[float]): A feature vector
    Returns:
        List[float]
    """
    arr_new = arr.copy()
    global training_min
    global training_max
    training_min = np.min(xTrain, axis=0)
    training_max = np.max(xTrain, axis=0)
    for j in range(arr.shape[1]):
        minVal = min(arr[:,j])
        maxVal = max(arr[:,j])
        arr_new[:,j] =  (arr_new[:,j] - minVal) / (maxVal - minVal)
    return arr_new
# alternative:
def normalize(X: List[float]) -> List[float]:
    """Normalizes values in a vector

    This function standardizes a feature vector by setting the mean to 0 and the standard deviation to 1. This is the
    second normalization function available.

    Args:
        X (List[float]): A feature vector
    Returns:
        List[float]
    """
    global training_mean
    global training_std
    x_new = X
    training_mean = np.mean(X, axis = 0)
    training_std = np.std(X, axis = 0)
    return (x_new - np.mean(x_new, axis = 0)) / np.std(x_new, axis = 0)




def showDependency(predictions, test, stat, my_categories):
    """Plots the actual values vs predictions for a given stat

    Args:
        predictions (List): The list of predictions generated by the model
        test ():
        stat ():
        my_categories():
    Returns:
        None
    """

    difference = test[:,my_categories.index(stat)]
    plt.scatter(difference, predictions)
    plt.ylabel('Probability of Team 1 Win')
    plt.xlabel(stat + ' Difference (Team 1 - Team 2)')
    plt.show()



def showFeatureImportance(my_categories):
    """Plots feature importance from model

    Args:
        my_categories (List[str]): A list of features to check the importance of
    Returns:
        None
    """

    fx_imp = pd.Series(model.feature_importances_, index=my_categories)
    fx_imp /= fx_imp.max()
    fx_imp.sort_values(ascending=True)
    fx_imp.plot(kind='barh')

def predictGame(team_1_vector: List[float], team_2_vector: List[float], home: int) -> float:
    """Runs a matchup vector through the trained model and outputs a probability that the first team wins

    Args:
        team_1_vector (List[float]): The team vector of the first team
        team_2_vector (List[float]): The team vector of the second team
        home (int): If the first team is home, this is 1. Away is -1, and neutral is 0
    Returns:
        float
    """
    global training_min
    global training_max
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    diff_flip = [a - b for a, b in zip(team_2_vector,team_1_vector)]
    diff.append(float(home))
    diff_flip.append(float(home))
    diff = (diff-training_min)/(training_max-training_min)
    diff_flip = (diff_flip-training_min)/(training_max-training_min)

    probability_team_1 = (model.predict_proba([diff])[0][1]+model.predict_proba([diff_flip])[0][0])/2

    return probability_team_1
    #return model.predict_proba([diff])



#read in data

#read compact data in as DataFrame
reg_season_compact_pd = pd.read_csv(r'files/MRegularSeasonCompactResults.csv')
print("Regular Season Compact\n\n",reg_season_compact_pd.head(),)

#read detailed data in as DataFrame
reg_season_detailed_pd = pd.read_csv('files/MRegularSeasonDetailedResults.csv')

pd.set_option('display.max_columns',None)
print("Example of a game\n\n",reg_season_detailed_pd.loc[1])

#read list of teams
teams_pd = pd.read_csv('files/Teams.csv')

#make a list of teams
teamList = teams_pd['Team_Name'].tolist()
print("These are a sample of the teams\n\n",teams_pd.tail())

#read tourney compact results
tourney_compact_pd = pd.read_csv('files/MNCAATourneyCompactResults.csv')
print("These are a sample of the compact tourney results\n\n",tourney_compact_pd.head())

#read tourney detailed results
tourney_detailed_pd = pd.read_csv('files/MNCAATourneyDetailedResults.csv')
print("These are a sample of the detailed tourney results\n\n",tourney_detailed_pd.head())

# read tourney seeds
tourney_seeds_pd = pd.read_csv('files/MNCAATourneySeeds.csv')
print("These are a sample of the tourney seeds\n\n",tourney_seeds_pd.head())

#read tourney slots
tourney_slots_pd = pd.read_csv('files/MNCAATourneySlots.csv')
print("These are a sample of the tourney slots\n\n",tourney_slots_pd.head())

#read conference info
conference_pd = pd.read_csv('files/Conference.csv')
print("This is a sample of the conference information\n\n",conference_pd.head())

#read tourney results
tourney_results_pd = pd.read_csv('files/TourneyResults.csv')
print("These are a sample of the tourney results\n\n",tourney_results_pd.head())
NCAAChampionsList = tourney_results_pd['NCAA Champion'].tolist()
logging.info("Data read successfully")

#add possession columns to dataframes
reg_season_detailed_pd = add_poss_stats(reg_season_detailed_pd)
print("This is the new reg season dataframe with all the per possession columns added\n\n",reg_season_detailed_pd.head())

tourney_detailed_pd = add_poss_stats(tourney_detailed_pd)
print("This is the new tourney results dataframe with all the per possession columns added\n\n",tourney_detailed_pd.head())

#create lists of teams in major conferences
listACCteams = ['North Carolina','Virginia','Florida St','Louisville','Notre Dame','Syracuse','Duke','Virginia Tech','Georgia Tech','Miami','Wake Forest','Clemson','NC State','Boston College','Pittsburgh']
listPac12teams = ['Arizona','Oregon','UCLA','California','USC','Utah','Washington St','Stanford','Arizona St','Colorado','Washington','Oregon St']
listSECteams = ['Kentucky','South Carolina','Florida','Arkansas','Alabama','Tennessee','Mississippi St','Georgia','Ole Miss','Vanderbilt','Auburn','Texas A&M','LSU','Missouri']
listBig10teams = ['Maryland','Wisconsin','Purdue','Northwestern','Michigan St','Indiana','Iowa','Michigan','Penn St','Nebraska','Minnesota','Illinois','Ohio St','Rutgers']
listBig12teams = ['Kansas','Baylor','West Virginia','Iowa St','TCU','Kansas St','Texas Tech','Oklahoma St','Texas','Oklahoma']
listBigEastteams = ['Butler','Creighton','DePaul','Georgetown','Marquette','Providence','Seton Hall','St John\'s','Villanova','Xavier']


# TODO this is unused
# getListForURL(teamList)



logging.info("Reading 2022 data")

# read this year's data. The code actually pulls this year's data from reg season detailed
this_year_pd = pd.read_csv("files/2022.csv")
handleDifferentCSV(this_year_pd)
for i in range(68):
    this_year_pd["Rk"][i]=int(getTeamID(this_year_pd["School"][i]))
print("This years data is: \n\n",this_year_pd)
logging.info("2022 data read successfully")


#test the functions to this point
print("The vector for teamID 1103 in 2022 is ",getSeasonData(1103,2022))

#get kentucky vector from 2021
kentucky_id = teams_pd[teams_pd['Team_Name'] == 'Kentucky'].values[0][0]
print("The vector for Kentucky in 2021 is ",getSeasonData(kentucky_id, 2021))




#test comparison of two teams in 2022
kansas_id = teams_pd[teams_pd['Team_Name'] == 'Kansas'].values[0][0]
print("The vector for teamIDs 1234 and 1242 in 2022 is ",compareTwoTeams(1234, 1242, 2022))


#TODO experiment with the range of years which leads to the best results
def get_x_and_y(load_model):
    if load_model == False:
        years_to_train = range(2003,2022)
        logging.info("Creating training set")
        xTrain, yTrain = createTrainingSet(years_to_train)
        logging.info("Training set created")
        np.save('xTrain', xTrain)
        np.save('yTrain', yTrain)
    else:


        # import numpy as np
        # xTrain = np.load('Data/March-Madness-2017-master/PrecomputedMatrices/xTrain.npy')
        # yTrain = np.load('Data/March-Madness-2017-master/PrecomputedMatrices/yTrain.npy')

        xTrain = np.load('xTrain.npy')
        yTrain = np.load('yTrain.npy')
    return xTrain, yTrain

training_data = get_x_and_y(True)
xTrain = training_data[0]
yTrain = training_data[1]
xTrainNorm = normalizeInput(xTrain)
print("xTrain shape: ",xTrain.shape,"\nyTrain shape: ",yTrain.shape)

# These are the different models I tried. Simply uncomment the model that you want to try.
# TODO utilize cross validation
# models = [tree.DecisionTreeClassifier(),AdaBoostClassifier(n_estimators=100),RandomForestClassifier(n_estimators=64),KNeighborsClassifier(n_neighbors=39)]
# model = tree.DecisionTreeClassifier()
# model = tree.DecisionTreeRegressor()
# model = linear_model.LogisticRegression()
# model = linear_model.LinearRegression()
# model = linear_model.BayesianRidge()
# model = linear_model.Lasso()
# model = svm.SVC()
# model = svm.SVR()
# model = linear_model.Ridge(alpha = 0.5)
# model = AdaBoostClassifier(n_estimators=100)
# model = GradientBoostingClassifier(n_estimators=100)
# model = [GradientBoostingRegressor(n_estimators=100, max_depth=5)]
model = RandomForestClassifier(n_jobs=-1,bootstrap=False, max_depth=10, max_features='auto', min_samples_leaf=12,
                               min_samples_split=15, n_estimators=50)
# model = KNeighborsClassifier(n_neighbors=39)
# neuralNetwork(10)
# model = VotingClassifier(estimators=[('GBR', model1), ('BR', model2), ('KNN', model3)], voting='soft')
# model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=0.1)





# categories=['Wins','PPG','PPGA','PowerConf','3PG', 'APG','TOP','Conference Champ','Tourney Conference Champ',
# 'Seed','SOS','SRS', 'RPG', 'SPG', 'Tourney Appearances','National Championships','totalPoss',
# 'totalfgmPerPoss', 'totalfgaPerPoss', 'totalfgm3PerPoss', 'totalfga3PerPoss', 'totalftmPerPoss',
# 'totalftaPerPoss', 'totalorPerPoss', 'totaldrPerPoss', 'totalastPerPoss', 'totaltoPerPoss',
# 'totalstlPerPoss', 'totalblkPerPoss', 'totalpfPerPoss','Location']


# model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
# model = linear_model.Ridge(alpha = 0.5)
# TODO utilize the categories or delete them
# categories = ['sos', 'srs', 'totalPoss', 'totalfgmPerPoss', 'totalfgaPerPoss', 'totalfgm3PerPoss', 'totalfga3PerPoss',
#               'totalftmPerPoss',
#               'totalftaPerPoss', 'totalorPerPoss', 'totaldrPerPoss', 'totalastPerPoss', 'totaltoPerPoss',
#               'totalstlPerPoss', 'totalblkPerPoss', 'totalpfPerPoss', 'Location']
# define the parameter grid
# param_grid = {'bootstrap': [False],
#  'max_depth': [10],
#  'max_features': ['auto'],
#  'min_samples_leaf': [8,9,10,11,12,13,14,15,16],
#  'min_samples_split': [7],
#  'n_estimators': [50]}

# import time
# ts = time.time()
# logging.info("Started grid search")
# # create a grid search object
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring = "f1_micro", verbose=2,n_jobs=-1)
#
# # fit the grid search object to the data
# grid_search.fit(xTrain, yTrain)
# logging.info(f"Finished grid search in {time.time()-ts}")

# print the best parameters
# print(grid_search.best_params_)
# After experimenting, {'bootstrap': False, max_depth': 10, 'max_features': auto,
# 'min_samples_leaf':13 , 'min_samples_split':7 , 'n_estimators': 50}
# grid_search.best_score_

# from sklearn.feature_selection import RFE
# rfe_selector = RFE(estimator=RandomForestClassifier(n_jobs=-1,bootstrap=False, max_depth=10, max_features='auto',
#                                                     min_samples_leaf=12, min_samples_split=15, n_estimators=50),
#                    step = 1)
# rfe_selector.fit(xTrain, yTrain)
# rfe_selector.get_support() #new_vector = [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id),
# #                                         avgAssists, avgTurnovers, tournamentSeed, getTourneyAppearances(team_id),
# #                                         totalPoss, totalfgmPerPoss, totalftmPerPoss, totaldrPerPoss, totalastPerPoss]

# accuracy = {}
#
# _scoring = ['accuracy', 'precision', 'recall', 'f1']
#
# results = cross_validate(estimator=model,
#                            X=xTrain,
#                            y=yTrain,
#                            cv=5,
#                            scoring=_scoring,
#                            return_train_score=True)
# accuracy[type(model).__name__+"trainf1"]=results['train_f1']
# accuracy[type(model).__name__+"testf1"]=results['test_f1']
# accuracy[type(model).__name__+"train_precision"]=results['train_precision']
# accuracy[type(model).__name__+"test_precision"]=results['test_precision']
# accuracy[type(model).__name__+"train_recall"]=results['train_recall']
# accuracy[type(model).__name__+"test_recall"]=results['test_recall']
#
# print(f"Random Forest's f1 score is {accuracy['RandomForestClassifiertrainf1'].mean()}")
# print(f"K Neighbors f1 score is {accuracy['KNeighborsClassifiertrainf1'].mean()}")
# print(f"Logistic Regression f1 score is {accuracy['LogisticRegressiontrainf1'].mean()}")
# print(f"Gradient Boost f1 score is {accuracy['GradientBoostingClassifiertrainf1'].mean()}")
#
# print(f"Random Forest's recall score is {accuracy['RandomForestClassifiertrain_recall'].mean()}")
# print(f"K Neighbors recall score is {accuracy['KNeighborsClassifiertrain_recall'].mean()}")
# print(f"Logistic Regression recall score is {accuracy['LogisticRegressiontrain_recall'].mean()}")
# print(f"Gradient Boost recall score is {accuracy['GradientBoostingClassifiertrain_recall'].mean()}")
#
# print(f"Random Forest's precision score is {accuracy['RandomForestClassifiertrain_precision'].mean()}")
# print(f"K Neighbors precision score is {accuracy['KNeighborsClassifiertrain_precision'].mean()}")
# print(f"Logistic Regression precision score is {accuracy['LogisticRegressiontrain_precision'].mean()}")
# print(f"Gradient Boost precision score is {accuracy['GradientBoostingClassifiertrain_precision'].mean()}")

# x_Train,y_Train,x_Test,y_Test = train_test_split()
# logging.info("Iterating through models")
# for i in range(10):
#     X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
#     results = model.fit(X_train, Y_train)
#     preds = model.predict(X_test)
#
#     preds[preds < .5] = 0
#     preds[preds >= .5] = 1
#     #TODO try different accuracy measure
#     accuracy.append(np.mean(preds == Y_test))
#     # accuracy.append(np.mean(predictions == Y_test))
#     print("n_estimators - ", 100, " max_depth - ", 5, "Finished iteration:", i, "The accuracy is",
#           sum(accuracy) / len(accuracy))
# print("n_estimators - ", 100, " max_depth - ", 5, "The accuracy is", sum(accuracy) / len(accuracy))



# showFeatureImportance(['Wins','PPG','PPGA','PowerConf','3PG', 'APG','TOP','Conference Champ','Tourney Conference Champ',
#              'Seed', 'RPG', 'SPG', 'Tourney Appearances','National Championships','totalPoss',
#              'totalfgmPerPoss', 'totalfgaPerPoss', 'totalfgm3PerPoss', 'totalfga3PerPoss', 'totalftmPerPoss',
#              'totalftaPerPoss', 'totalorPerPoss', 'totaldrPerPoss', 'totalastPerPoss', 'totaltoPerPoss',
#              'totalstlPerPoss', 'totalblkPerPoss', 'totalpfPerPoss','Location'])

#showFeatureImportance(['Wins','SOS','SRS','location'])


model.fit(xTrainNorm,yTrain)



# This can be used to predict 2022 games
team1_name = "Creighton"
team2_name = "San Diego St"
team1_vector = get2022Data(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0])
team2_vector = get2022Data(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0])

print ('Probability that ' + team1_name + ' wins:', predictGame(team1_vector, team2_vector, 0))




# TODO uncomment to create submission for Kaggle
submission=pd.read_csv("C:/Users/Lenovo/Documents/MDataFiles_Stage2/MSampleSubmissionStage2.csv")
submission
#
#
#
logging.info("Creating submission csv")
preds=[]
for i in range(2278):
    vector1=get2022Data(int(submission.iloc[i][0][5:9]))
    vector2=get2022Data(int(submission.iloc[i][0][10:14]))
    pred=predictGame(vector1, vector2, 0)
    preds.append(pred)
submission["Pred"]=preds
#
#
#
submission.tail()


submission.to_csv("files/2022MarchMadnessKaggleLMR.csv")
logging.info("Submission csv created successfully")