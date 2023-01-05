import sklearn
import pandas as pd
import numpy as np
import itertools
from __future__ import division
import collections
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import tree
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD
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
%matplotlib inline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import urllib
from sklearn.svm import LinearSVC
from utils import *


#read in data

#read compact data in as DataFrame
reg_season_compact_pd = pd.read_csv(r'C:/Users/Lenovo/Documents/Data/MRegularSeasonCompactResults.csv')
print("Regular Season Compact\n\n",reg_season_compact_pd.head(),)

#read detailed data in as DataFrame
reg_season_detailed_pd = pd.read_csv('C:/Users/Lenovo/Documents/Data/MRegularSeasonDetailedResults.csv')

pd.set_option('display.max_columns',None)
print("Example of a game\n\n",reg_season_detailed_pd.loc[1])

#read list of teams
teams_pd = pd.read_csv('C:/Users/Lenovo/Documents/Data/Teams.csv')

#make a list of teams
teamList = teams_pd['Team_Name'].tolist()
print("These are a sample of the teams\n\n",teams_pd.tail())

#read tourney compact results
tourney_compact_pd = pd.read_csv('C:/Users/Lenovo/Documents/Data/MNCAATourneyCompactResults.csv')
print("These are a sample of the compact tourney results\n\n",tourney_compact_pd.head())

#read tourney detailed results
tourney_detailed_pd = pd.read_csv('C:/Users/Lenovo/Documents/Data/MNCAATourneyDetailedResults.csv')
print("These are a sample of the detailed tourney results\n\n",tourney_detailed_pd.head())

# read tourney seeds
tourney_seeds_pd = pd.read_csv('C:/Users/Lenovo/Documents/Data/MNCAATourneySeeds.csv')
print("These are a sample of the tourney seeds\n\n",tourney_seeds_pd.head())

#read tourney slots
tourney_slots_pd = pd.read_csv('C:/Users/Lenovo/Documents/Data/MNCAATourneySlots.csv')
print("These are a sample of the tourney slots\n\n",tourney_slots_pd.head())

#read conference info
conference_pd = pd.read_csv('C:/Users/Lenovo/Documents/Data/Conference.csv')
print("This is a sample of the conference information\n\n",conference_pd.head()

#read tourney results
tourney_results_pd = pd.read_csv('C:/Users/Lenovo/Documents/Data/TourneyResults.csv')
print("These are a sample of the tourney results\n\n",tourney_results_pd.head())
NCAAChampionsList = tourney_results_pd['NCAA Champion'].tolist()

def add_poss_stats(df):
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



def checkPower6Conference(team_id):
    teamName = teams_pd.values[team_id-1101][1]
    if (teamName in listACCteams or teamName in listBig10teams or teamName in listBig12teams
       or teamName in listSECteams or teamName in listPac12teams or teamName in listBigEastteams):
        return 1
    else:
        return 0

def getTeamID(name):
    return teams_pd[teams_pd['Team_Name'] == name].values[0][0]


def getTeamName(team_id):
    return teams_pd[teams_pd['Team_Id'] == team_id].values[0][1]


def getNumChampionships(team_id):
    name = getTeamName(team_id)
    return NCAAChampionsList.count(name)


def getListForURL(team_list):
    team_list = [x.lower() for x in team_list]
    team_list = [t.replace(' ', '-') for t in team_list]
    team_list = [t.replace('st', 'state') for t in team_list]
    team_list = [t.replace('northern-dakota', 'north-dakota') for t in team_list]
    team_list = [t.replace('nc-', 'north-carolina-') for t in team_list]
    team_list = [t.replace('fl-', 'florida-') for t in team_list]
    team_list = [t.replace('ga-', 'georgia-') for t in team_list]
    team_list = [t.replace('lsu', 'louisiana-state') for t in team_list]
    team_list = [t.replace('maristate', 'marist') for t in team_list]
    team_list = [t.replace('stateate', 'state') for t in team_list]
    team_list = [t.replace('northernorthern', 'northern') for t in team_list]
    team_list = [t.replace('usc', 'southern-california') for t in team_list]
    base = 'http://www.sports-reference.com/cbb/schools/'
    for team in team_list:
        url = base + team + '/'
getListForURL(teamList)



# Function for handling the annoying cases of Florida and FL, as well as State and St
def handleCases(arr):
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



def checkConferenceChamp(team_id, year):
    year_conf_pd = conference_pd[conference_pd['Year'] == year]
    champs = year_conf_pd['Regular Season Champ'].tolist()
    # For handling cases where there is more than one champion
    champs_separated = [words for segments in champs for words in segments.split()]
    name = getTeamName(team_id)
    champs_separated = handleCases(champs_separated)
    if (name in champs_separated):
        return 1
    else:
        return 0



def checkConferenceTourneyChamp(team_id, year):
    year_conf_pd = conference_pd[conference_pd['Year'] == year]
    champs = year_conf_pd['Tournament Champ'].tolist()
    name = getTeamName(team_id)
    if (name in champs):
        return 1
    else:
        return 0



def getTourneyAppearances(team_id):
    return len(tourney_seeds_pd[tourney_seeds_pd['TeamID'] == team_id].index)



def handleDifferentCSV(df):
    # The stats CSV is a lit different in terms of naming so below is just some data cleaning
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


def season_totals(stat,df,team_id):
    total = stat['WScore'].sum()
    gamesLost = df[df.LTeamID == team_id]
    totalGames = stat.append(gamesLost)
    numGames = len(totalGames.index)
    total += gamesLost['LScore'].sum()
    return total, numGames

def getSeasonData(team_id, year):
    # The data frame below holds stats for every single game in the given year
    year_data_pd = reg_season_detailed_pd[reg_season_detailed_pd['Season'] == year]

    gamesWon = year_data_pd[year_data_pd.WTeamID == team_id]
    gamesLost = year_data_pd[year_data_pd.LTeamID == team_id]

    # Finding number of points in season
    totalPointsScored = season_totals('WScore',year_data_pd,team_id)[0]
    numGames = season_totals('WScore',year_data_pd,team_id)[1]

    # Finding number of possessions in season
    totalPoss = season_totals('Wposs', year_data_pd, team_id)[0]

    # Finding number of fgm in season
    totalFgm = season_totals('WFGM', year_data_pd, team_id)[0]

    # Finding number of fga in season
    totalFga = season_totals('WFGA', year_data_pd, team_id)[0]

    # Finding number of fgm3 in season
    totalFgm3 = season_totals('WFGM3', year_data_pd, team_id)[0]

    # Finding number of fga3 in season
    totalFga3 = season_totals('WFGA3', year_data_pd, team_id)[0]

    # Finding number of ftm in season
    totalFtm = season_totals('WFTM', year_data_pd, team_id)[0]

    # Finding number of fta in season
    totalFta = season_totals('WFTA', year_data_pd, team_id)[0]

    # Finding number of or in season
    totalOr = season_totals('WOR', year_data_pd, team_id)[0]

    # Finding number of dr in season
    totalDr = season_totals('WDR', year_data_pd, team_id)[0]

    totalReb = totalOr + totalDr

    # Finding number of blk in season
    totalBlk = season_totals('WBlk', year_data_pd, team_id)[0]

    # Finding number of pf in season
    totalPf = season_totals('WPF', year_data_pd, team_id)[0]

    # Finding number of to in season
    totalTo = season_totals('WTO',year_data_pd,team_id)[0]

    # Finding number of ast in season
    totalAst = season_totals('WAst', year_data_pd, team_id)[0]

    # Finding number of Stl in season
    totalStl = season_totals('WStl', year_data_pd, team_id)[0]

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
    if (len(seed.index) != 0):
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
    return [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id), avg3sMade, avgAssists,
            avgTurnovers,
            checkConferenceChamp(team_id, year), checkConferenceTourneyChamp(team_id, year), tournamentSeed,
            avgRebounds, avgSteals, getTourneyAppearances(team_id), getNumChampionships(team_id), totalPoss,
            totalfgmPerPoss, totalfgaPerPoss, totalfgm3PerPoss, totalfga3PerPoss, totalftmPerPoss, totalftaPerPoss,
            totalorPerPoss, totaldrPerPoss, totalastPerPoss, totaltoPerPoss, totalstlPerPoss, totalblkPerPoss,
            totalpfPerPoss]

    # return [sos,srs,totalPoss, totalfgmPerPoss, totalfgaPerPoss, totalfgm3PerPoss, totalfga3PerPoss, totalftmPerPoss, totalftaPerPoss,
    #       totalorPerPoss, totaldrPerPoss, totalastPerPoss, totaltoPerPoss, totalstlPerPoss, totalblkPerPoss, totalpfPerPoss]



this_year_pd = pd.read_csv("C:/Users/Lenovo/Documents/Data/2022.csv")
handleDifferentCSV(this_year_pd)
for i in range(68):
    this_year_pd["Rk"][i]=int(getTeamID(this_year_pd["School"][i]))
this_year_pd


def get2022Data(team_id):
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

    return [int(numWins.iloc[0]), float(avgPointsScored), float(avgPointsAllowed), int(checkPower6Conference(team_id)),
            float(avg3sMade), float(avgAssists), float(avgTurnovers),
            int(reg_conf_champ), int(conf_tourn_champ), int(tournamentSeed),
            float(avgRebounds), float(avgSteals), int(getTourneyAppearances(team_id)),
            int(getNumChampionships(team_id)), totalPoss.iloc[0],
            totalfgmPerPoss.iloc[0], totalfgaPerPoss.iloc[0], totalfgm3PerPoss.iloc[0], totalfga3PerPoss.iloc[0],
            totalftmPerPoss.iloc[0], totalftaPerPoss.iloc[0],
            totalorPerPoss.iloc[0], totaldrPerPoss.iloc[0], totalastPerPoss.iloc[0], totaltoPerPoss.iloc[0],
            totalstlPerPoss.iloc[0], totalblkPerPoss.iloc[0], totalpfPerPoss.iloc[0]]



getSeasonData(1103,2022)

this_year_pd["School"][36]

kentucky_id = teams_pd[teams_pd['Team_Name'] == 'Kentucky'].values[0][0]
getSeasonData(kentucky_id, 2021)



def compareTwoTeams(id_1, id_2, year):
    if year==2022:
        team_1 = get2022Data(id_1)
        team_2 = get2022Data(id_2)
    else:
        team_1 = getSeasonData(id_1, year)
        team_2 = getSeasonData(id_2, year)
    diff = [a - b for a, b in zip(team_1, team_2)]
    return diff



kansas_id = teams_pd[teams_pd['Team_Name'] == 'Kansas'].values[0][0]
compareTwoTeams(1234, 1242, 2022)



def createSeasonDict(year):
    seasonDictionary = collections.defaultdict(list)
    for team in teamList:
        team_id = teams_pd[teams_pd['Team_Name'] == team].values[0][0]
        team_vector = getSeasonData(team_id, year)
        seasonDictionary[team_id] = team_vector
    return seasonDictionary



def getHomeStat(row):
    if (row == 'H'):
        home = 1
    if (row == 'A'):
        home = -1
    if (row == 'N'):
        home = 0
    return home



def createTrainingSet(years):
    totalNumGames = 0
    for year in years:
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
        totalNumGames += len(season.index)
        tourney = tourney_compact_pd[tourney_compact_pd['Season'] == year]
        totalNumGames += len(tourney.index)
    numFeatures = len(getSeasonData(1181,2012)) #Just choosing a random team and seeing the dimensionality of the vector
    xTrain = np.zeros(( totalNumGames, numFeatures + 1))
    yTrain = np.zeros(( totalNumGames ))
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
        xTrain[indexCounter:numGamesInSeason+indexCounter] = xTrainSeason
        yTrain[indexCounter:numGamesInSeason+indexCounter] = yTrainSeason
        indexCounter += numGamesInSeason
    return xTrain, yTrain



def normalizeInput(arr):
    for i in range(arr.shape[1]):
        minVal = min(arr[:,i])
        maxVal = max(arr[:,i])
        arr[:,i] =  (arr[:,i] - minVal) / (maxVal - minVal)
    return arr
# alternative:
def normalize(X):
    return (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)



years = range(1993,2022)
xTrain, yTrain = createTrainingSet(years)
np.save('xTrain', xTrain)
np.save('yTrain', yTrain)



# import numpy as np
# xTrain = np.load('Data/March-Madness-2017-master/PrecomputedMatrices/xTrain.npy')
# yTrain = np.load('Data/March-Madness-2017-master/PrecomputedMatrices/yTrain.npy')



xTrain.shape
yTrain.shape



# These are the different models I tried. Simply uncomment the model that you want to try.

model = [tree.DecisionTreeClassifier(), tree.DecisionTreeRegressor(), linear_model.LogisticRegression(),
         linear_model.Lasso(), linear_model.Ridge(alpha = 0.5),
         AdaBoostClassifier(n_estimators=100), GradientBoostingClassifier(n_estimators=100), GradientBoostingRegressor(n_estimators=100, max_depth=5),
        RandomForestClassifier(n_estimators=64), KNeighborsClassifier(n_neighbors=39), LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=0.1)]
#model = tree.DecisionTreeRegressor()
#model = linear_model.LogisticRegression()
#model = linear_model.BayesianRidge()
#model = linear_model.Lasso()
#model = svm.SVC()
#model = svm.SVR()
#model = linear_model.Ridge(alpha = 0.5)
#model = AdaBoostClassifier(n_estimators=100)
#model = GradientBoostingClassifier(n_estimators=100)
#model = [GradientBoostingRegressor(n_estimators=100, max_depth=5)]
#model = RandomForestClassifier(n_estimators=64)
#model = KNeighborsClassifier(n_neighbors=39)
#neuralNetwork(10)
#model = VotingClassifier(estimators=[('GBR', model1), ('BR', model2), ('KNN', model3)], voting='soft')
#model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=0.1)



def showDependency(predictions, test, stat, my_categories):
    difference = test[:,my_categories.index(stat)]
    plt.scatter(difference, predictions)
    plt.ylabel('Probability of Team 1 Win')
    plt.xlabel(stat + ' Difference (Team 1 - Team 2)')
    plt.show()



def showFeatureImportance(my_categories):
    fx_imp = pd.Series(model.feature_importances_, index=my_categories)
    fx_imp /= fx_imp.max()
    fx_imp.sort_values(ascending=True)
    fx_imp.plot(kind='barh')


# categories=['Wins','PPG','PPGA','PowerConf','3PG', 'APG','TOP','Conference Champ','Tourney Conference Champ',
# 'Seed','SOS','SRS', 'RPG', 'SPG', 'Tourney Appearances','National Championships','totalPoss',
# 'totalfgmPerPoss', 'totalfgaPerPoss', 'totalfgm3PerPoss', 'totalfga3PerPoss', 'totalftmPerPoss',
# 'totalftaPerPoss', 'totalorPerPoss', 'totaldrPerPoss', 'totalastPerPoss', 'totaltoPerPoss',
# 'totalstlPerPoss', 'totalblkPerPoss', 'totalpfPerPoss','Location']


model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
# model = linear_model.Ridge(alpha = 0.5)
categories = ['sos', 'srs', 'totalPoss', 'totalfgmPerPoss', 'totalfgaPerPoss', 'totalfgm3PerPoss', 'totalfga3PerPoss',
              'totalftmPerPoss',
              'totalftaPerPoss', 'totalorPerPoss', 'totaldrPerPoss', 'totalastPerPoss', 'totaltoPerPoss',
              'totalstlPerPoss', 'totalblkPerPoss', 'totalpfPerPoss', 'Location']
accuracy = []

for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
    results = model.fit(X_train, Y_train)
    preds = model.predict(X_test)

    preds[preds < .5] = 0
    preds[preds >= .5] = 1
    accuracy.append(np.mean(preds == Y_test))
    # accuracy.append(np.mean(predictions == Y_test))
    print("n_estimators - ", 100, " max_depth - ", 5, "Finished iteration:", i, "The accuracy is",
          sum(accuracy) / len(accuracy))
print("n_estimators - ", 100, " max_depth - ", 5, "The accuracy is", sum(accuracy) / len(accuracy))



showFeatureImportance(['Wins','PPG','PPGA','PowerConf','3PG', 'APG','TOP','Conference Champ','Tourney Conference Champ',
             'Seed', 'RPG', 'SPG', 'Tourney Appearances','National Championships','totalPoss',
             'totalfgmPerPoss', 'totalfgaPerPoss', 'totalfgm3PerPoss', 'totalfga3PerPoss', 'totalftmPerPoss',
             'totalftaPerPoss', 'totalorPerPoss', 'totaldrPerPoss', 'totalastPerPoss', 'totaltoPerPoss',
             'totalstlPerPoss', 'totalblkPerPoss', 'totalpfPerPoss','Location'])

#showFeatureImportance(['Wins','SOS','SRS','location'])



def predictGame(team_1_vector, team_2_vector, home):
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    diff.append(home)
    return model.predict([diff])
    #return model.predict_proba([diff])



# This was the national championship matchup last year
team1_name = "Duke"
team2_name = "Arizona"
team1_vector = get2022Data(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0])
team2_vector = get2022Data(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0])
print ('Probability that ' + team1_name + ' wins:', predictGame(team1_vector, team2_vector, 0)[0])



# This was the national championship matchup last year
team1_name = "Arizona"
team2_name = "Kansas"
team1_vector = get2022Data(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0])
team2_vector = get2022Data(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0])
print ('Probability that ' + team1_name + ' wins:', predictGame(team1_vector, team2_vector, 0)[0])



# This was the national championship matchup last year
team1_name = "Tennessee"
team2_name = "Villanova"
team1_vector = get2022Data(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0])
team2_vector = get2022Data(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0])
print ('Probability that ' + team1_name + ' wins:', predictGame(team1_vector, team2_vector, 0)[0])



submission=pd.read_csv("C:/Users/Lenovo/Documents/MDataFiles_Stage2/MSampleSubmissionStage2.csv")
submission



preds=[]
for i in range(2278):
    vector1=get2022Data(int(submission.iloc[i][0][5:9]))
    vector2=get2022Data(int(submission.iloc[i][0][10:14]))
    pred=predictGame(vector1, vector2, 0)[0]
    preds.append(pred)
submission["Pred"]=preds



submission.tail()



# submission.to_csv("C:/Users/Lenovo/Documents/2022MarchMadnessKaggleLMR.csv")