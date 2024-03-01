from __future__ import annotations

import pickle

import dill

import classifier
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
from sklearn.model_selection import cross_val_score
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
import xgboost as xgb

logging.basicConfig(level=logging.DEBUG)

training_mean = 0
training_std = 0
training_min = 0
training_max = 0


# TODO add logging
# TODO add error catching
# TODO add files to github
class Data:
    def __init__(self):
        self.listACCteams = []
        self.listBig10teams = []
        self.listBig12teams = []
        self.listSECteams = []
        self.listPac12teams = []
        self.listBig10teams = []
        self.listBigEastteams = []
        self.rfecv = None
        self.gender = ""

    def read_data(self, gender):
        def filter_by_season(row, tourney_teams):
            return (row['WTeamID'] in tourney_teams) and (row['LTeamID'] in tourney_teams)

        # read in data
        if gender == "men":
            # read compact data in as DataFrame
            self.reg_season_compact_pd = pd.read_csv(r'files/MRegularSeasonCompactResults.csv')
            print("Regular Season Compact\n\n", self.reg_season_compact_pd.head(), )

            # read detailed data in as DataFrame
            self.reg_season_detailed_pd = pd.read_csv('files/MRegularSeasonDetailedResults.csv')

            pd.set_option('display.max_columns', None)
            print("Example of a game\n\n", self.reg_season_detailed_pd.loc[1])

            # read list of teams
            self.teams_pd = pd.read_csv('files/MTeams.csv')

            # make a list of teams
            self.teamList = self.teams_pd['TeamName'].tolist()
            print("These are a sample of the teams\n\n", self.teams_pd.tail())

            # read tourney compact results
            self.tourney_compact_pd = pd.read_csv('files/MNCAATourneyCompactResults.csv')
            print("These are a sample of the compact tourney results\n\n", self.tourney_compact_pd.head())

            # read tourney detailed results
            self.tourney_detailed_pd = pd.read_csv('files/MNCAATourneyDetailedResults.csv')
            print("These are a sample of the detailed tourney results\n\n", self.tourney_detailed_pd.head())

            # read tourney seeds
            self.tourney_seeds_pd = pd.read_csv('files/MNCAATourneySeeds.csv')
            print("These are a sample of the tourney seeds\n\n", self.tourney_seeds_pd.head())
            self.read_ordinals()
            merged_df = pd.merge(self.reg_season_detailed_pd, self.ordinals, how='left',
                                 left_on=['Season', 'DayNum', 'WTeamID'],
                                 right_on=['Season', 'RankingDayNum', 'TeamID'])
            merged_df = pd.merge(merged_df, self.ordinals, how='left', left_on=['Season', 'DayNum', 'LTeamID'],
                                 right_on=['Season', 'RankingDayNum', 'TeamID'], suffixes=('_W', '_L'))
            self.filtered_regular = merged_df[(merged_df['OrdinalRank_W'] < 125) | (merged_df['OrdinalRank_L'] < 125)]
            # self.filtered_regular = self.reg_season_detailed_pd[self.reg_season_detailed_pd.apply(
            #     lambda row: filter_by_season(row, set(self.tourney_seeds_pd[self.tourney_seeds_pd['Season'] == row['Season']]['TeamID'])), axis=1)]

            # read tourney slots
            self.tourney_slots_pd = pd.read_csv('files/MNCAATourneySlots.csv')
            print("These are a sample of the tourney slots\n\n", self.tourney_slots_pd.head())

            # read conference info
            self.conference_pd = pd.read_csv('files/Conferences.csv')
            print("This is a sample of the conference information\n\n", self.conference_pd.head())

            # read tourney results
            # self.tourney_results_pd = pd.read_csv('files/TourneyResults.csv')
            # print("These are a sample of the tourney results\n\n", self.tourney_results_pd.head())
            # NCAAChampionsList = self.tourney_results_pd['NCAA Champion'].tolist()
            # logging.info("Data read successfully")

            # add possession columns to dataframes
            self.filtered_regular = self.add_poss_stats(self.filtered_regular)
            print("This is the new reg season dataframe with all the per possession columns added\n\n",
                  self.filtered_regular.head())

            self.tourney_detailed_pd = self.add_poss_stats(self.tourney_detailed_pd)
            print("This is the new tourney results dataframe with all the per possession columns added\n\n",
                  self.tourney_detailed_pd.head())
            return
        elif gender == "women":
            # read compact data in as DataFrame
            self.reg_season_compact_pd = pd.read_csv(r'files/WRegularSeasonCompactResults.csv')
            print("Regular Season Compact\n\n", self.reg_season_compact_pd.head(), )

            # read detailed data in as DataFrame
            self.reg_season_detailed_pd = pd.read_csv('files/WRegularSeasonDetailedResults.csv')

            pd.set_option('display.max_columns', None)
            print("Example of a game\n\n", self.reg_season_detailed_pd.loc[1])

            # read list of teams
            self.teams_pd = pd.read_csv('files/WTeams.csv')

            # make a list of teams
            self.teamList = self.teams_pd['TeamName'].tolist()
            print("These are a sample of the teams\n\n", self.teams_pd.tail())

            # read tourney compact results
            self.tourney_compact_pd = pd.read_csv('files/WNCAATourneyCompactResults.csv')
            print("These are a sample of the compact tourney results\n\n", self.tourney_compact_pd.head())

            # read tourney detailed results
            self.tourney_detailed_pd = pd.read_csv('files/WNCAATourneyDetailedResults.csv')
            print("These are a sample of the detailed tourney results\n\n", self.tourney_detailed_pd.head())

            # read tourney seeds
            self.tourney_seeds_pd = pd.read_csv('files/WNCAATourneySeeds.csv')
            print("These are a sample of the tourney seeds\n\n", self.tourney_seeds_pd.head())

            self.filtered_regular = self.reg_season_detailed_pd[self.reg_season_detailed_pd.apply(
                lambda row: filter_by_season(row, set(
                    self.tourney_seeds_pd[self.tourney_seeds_pd['Season'] == row['Season']]['TeamID'])), axis=1)]

            # read tourney slots
            self.tourney_slots_pd = pd.read_csv('files/WNCAATourneySlots.csv')
            print("These are a sample of the tourney slots\n\n", self.tourney_slots_pd.head())

            # read conference info
            self.conference_pd = pd.read_csv('files/Conferences.csv')
            print("This is a sample of the conference information\n\n", self.conference_pd.head())

            # read tourney results
            # self.tourney_results_pd = pd.read_csv('files/TourneyResults.csv')
            # print("These are a sample of the tourney results\n\n", self.tourney_results_pd.head())
            # NCAAChampionsList = self.tourney_results_pd['NCAA Champion'].tolist()
            # logging.info("Data read successfully")

            # add possession columns to dataframes
            self.filtered_regular = self.add_poss_stats(self.filtered_regular)
            print("This is the new reg season dataframe with all the per possession columns added\n\n",
                  self.filtered_regular.head())

            self.tourney_detailed_pd = self.add_poss_stats(self.tourney_detailed_pd)
            print("This is the new tourney results dataframe with all the per possession columns added\n\n",
                  self.tourney_detailed_pd.head())
            return

    def read_ordinals(self):
        df = pd.read_csv('files/MMasseyOrdinals.csv')
        # Create a combination of all possible combinations
        all_combinations = pd.MultiIndex.from_product([df['Season'].unique(),
                                                       df['RankingDayNum'].unique(),
                                                       df['SystemName'].unique(),
                                                       df['TeamID'].unique()],
                                                      names=['Season', 'RankingDayNum', 'SystemName', 'TeamID'])

        # Convert to DataFrame
        all_combinations_df = pd.DataFrame(index=all_combinations).reset_index()

        # Merge with the original DataFrame to fill in missing values
        merged_df = pd.merge(all_combinations_df, df, how='left',
                             on=['Season', 'RankingDayNum', 'SystemName', 'TeamID'])

        # Fill in missing values with the previous value for each system
        merged_df['OrdinalRank'] = merged_df.groupby(['SystemName', 'TeamID'])['OrdinalRank'].fillna(method='ffill')

        # Calculate the average OrdinalRank across all systems for each TeamID on a specific day and season
        result_df = merged_df.groupby(['Season', 'RankingDayNum', 'TeamID'])['OrdinalRank'].mean().reset_index()
        self.ordinals = result_df
        return

    def add_poss_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds possession statistics to dataframe

        This calculates number of possessions and adds columns to indicate possessions and per possession stats

        Args:
            df (DataFrame): A DataFrame containing detailed NCAA game statistics
        Returns:
            DataFrame
        """

        # handle possessions for detailed tourney results

        # adds possession column
        df['Wposs'] = df['WFGA'] - df['WOR'] + df[
            'WTO'] + .44 * df['WFTA']
        df['Lposs'] = df['LFGA'] - df['LOR'] + df[
            'LTO'] + .44 * df['LFTA']

        # determines 1st and 3rd quantiles for possession numbers
        tourney_poss_quantiles = df['Wposs'].quantile([.25, .5, .75])
        tourney_poss_low = tourney_poss_quantiles.iloc[0]
        tourney_poss_mean = tourney_poss_quantiles.iloc[1]
        tourney_poss_high = tourney_poss_quantiles.iloc[2]

        # adds a regularized column for possessions
        df['Wposs_sc'] = (df['Wposs'] - tourney_poss_low) / (
                tourney_poss_high - tourney_poss_low)
        df['Lposs_sc'] = (df['Lposs'] - tourney_poss_low) / (
                tourney_poss_high - tourney_poss_low)

        # adds columns for stats per possession
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

    def checkPower6Conference(self, team_id: int) -> int:
        """Checks if a team is in a power conference

        This checks if a team is in oneof the top 6 larger conferenes in the NCAA it returns 1 if so and 0 if not

        Args:
            team_id (int): The integer id of a given team
        Returns:
            int
        """

        try:
            teamName = self.teams_pd.values[team_id - 1101][1]
        except:
            teamName = self.teams_pd.values[team_id - 3101][1]
        if (teamName in self.listACCteams or teamName in self.listBig10teams or teamName in self.listBig12teams
                or teamName in self.listSECteams or teamName in self.listPac12teams or teamName in self.listBigEastteams):
            return 1
        else:
            return 0

    def getTeamID(self, name: str) -> int:
        """Gets team id from name of school

        Args:
            name (str): The name of an NCAA school
        Returns:
            int
        """
        return self.teams_pd[self.teams_pd['TeamName'] == name].values[0][0]

    def getTeamName(self, team_id: int) -> str:
        """Gets team name from team id

        Args:
            team_id (int): Integer id of an NCAA team
        Returns:
            str
        """

        return self.teams_pd[self.teams_pd['TeamID'] == team_id].values[0][1]

    def getNumChampionships(self, team_id: int) -> int:
        """Gets the number of chapionships won by a team based on team id

        Args:
            team_id (int): Integer id of an NCAA team
        Returns:
            int
        """

        name = self.getTeamName(team_id)
        return self.NCAAChampionsList.count(name)

    # Function for handling the annoying cases of Florida and FL, as well as State and St
    def handleCases(self, arr: list[str]) -> list[str]:
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

    def checkConferenceChamp(self, team_id: int, year: int) -> int:
        """Checks if a team won their conference in a given year.

        If the team won their conference, the function returns 1. Otherwise, it returns 0.

        Args:
            team_id (int): Integer id of an NCAA team
            year (int): Full year value
        Returns:
            int
        """

        year_conf_pd = self.conference_pd[self.conference_pd['Year'] == year]
        champs = year_conf_pd['Regular Season Champ'].tolist()
        # For handling cases where there is more than one champion
        champs_separated = [words for segments in champs for words in segments.split()]
        name = self.getTeamName(team_id)
        champs_separated = self.handleCases(champs_separated)
        if name in champs_separated:
            return 1
        else:
            return 0

    def checkConferenceTourneyChamp(self, team_id: int, year: int) -> int:
        """Checks if a team won their conference championship game in a given year.

        If the team won their conference championship, the function returns 1. Otherwise it returns 0.

        Args:
            team_id (int): Integer id of an NCAA team
            year (int): Full year value
        Returns:
            int
        """

        year_conf_pd = self.conference_pd[self.conference_pd['Year'] == year]
        champs = year_conf_pd['Tournament Champ'].tolist()
        name = self.getTeamName(team_id)
        if name in champs:
            return 1
        else:
            return 0

    def getTourneyAppearances(self, team_id: int) -> int:
        """Checks how many times a team has played in the NCAA tournamnet.

        Args:
            team_id (int): Integer id of an NCAA team
        Returns:
            int
        """
        return len(self.tourney_seeds_pd[self.tourney_seeds_pd['TeamID'] == team_id].index)

    def handleDifferentCSV(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def season_totals(self, wstat: str, lstat: str, gamesWon: pd.DataFrame, df: pd.DataFrame, team_id: int) -> Tuple[
        int, int]:
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

    def getSeasonData(self, team_id: int, year: int, cleaned_features=False) -> List[float]:
        """Gets the full season data for a given team in a given year

        This function looks at a team in a given year and returns a list of statistics which becomes the team's vector for
        that season. This list of statistics can be edited and there are multiple iterations commented out in the code which
        have been used at various stages of testing.

        Args:
            team_id (int): Integer id of an NCAA team
            year (int): Full year value
            cleaned_features(Bool): Use full feature list or use pre selected features based on RFE selection.
                                    Default is True
        Returns:
            List[float]
        """

        # The data frame below holds stats for every single game in the given year
        year_data_pd = self.filtered_regular[self.filtered_regular['Season'] == year]
        # Finding number of points per game
        gamesWon = year_data_pd[year_data_pd.WTeamID == team_id]
        gamesLost = year_data_pd[year_data_pd.LTeamID == team_id]

        totalPointsScored = self.season_totals('WScore', 'LScore', gamesWon, year_data_pd, team_id)[0]
        numGames = self.season_totals('WScore', 'LScore', gamesWon, year_data_pd, team_id)[1]

        # Finding number of possessions in season
        totalPoss = self.season_totals('Wposs', 'Lposs', gamesWon, year_data_pd, team_id)[0]

        # Finding number of fgm in season
        totalFgm = self.season_totals('WFGM', 'LFGM', gamesWon, year_data_pd, team_id)[0]
        totalOppFGM = self.season_totals('LFGM', 'WFGM', gamesWon, year_data_pd, team_id)[0]

        # Finding number of fga in season
        totalFga = self.season_totals('WFGA', 'LFGA', gamesWon, year_data_pd, team_id)[0]
        totalOppFga = self.season_totals('LFGA', 'WFGA', gamesWon, year_data_pd, team_id)[0]

        # Finding number of fgm3 in season
        totalFgm3 = self.season_totals('WFGM3', 'LFGM3', gamesWon, year_data_pd, team_id)[0]
        totalOppFgm3 = self.season_totals('LFGM3', 'WFGM3', gamesWon, year_data_pd, team_id)[0]

        # Finding number of fga3 in season
        totalFga3 = self.season_totals('WFGA3', 'LFGA3', gamesWon, year_data_pd, team_id)[0]
        totalOppFga3 = self.season_totals('LFGA3', 'WFGA3', gamesWon, year_data_pd, team_id)[0]

        # Finding number of ftm in season
        totalFtm = self.season_totals('WFTM', 'LFTM', gamesWon, year_data_pd, team_id)[0]
        totalOppFtm = self.season_totals('LFTM', 'WFTM', gamesWon, year_data_pd, team_id)[0]

        # Finding number of fta in season
        totalFta = self.season_totals('WFTA', 'LFTA', gamesWon, year_data_pd, team_id)[0]
        totalOppFta = self.season_totals('LFTA', 'WFTA', gamesWon, year_data_pd, team_id)[0]

        # Finding number of or in season
        totalOr = self.season_totals('WOR', 'LOR', gamesWon, year_data_pd, team_id)[0]
        totalOppOr = self.season_totals('LOR', 'WOR', gamesWon, year_data_pd, team_id)[0]

        # Finding number of dr in season
        totalDr = self.season_totals('WDR', 'LDR', gamesWon, year_data_pd, team_id)[0]
        totalOppDr = self.season_totals('LDR', 'WDR', gamesWon, year_data_pd, team_id)[0]

        totalReb = totalOr + totalDr
        totalOppReb = totalOppOr + totalOppDr

        # Finding number of blk in season
        totalBlk = self.season_totals('WBlk', 'LBlk', gamesWon, year_data_pd, team_id)[0]
        totalOppBlk = self.season_totals('LBlk', 'WBlk', gamesWon, year_data_pd, team_id)[0]

        # Finding number of pf in season
        totalPf = self.season_totals('WPF', 'LPF', gamesWon, year_data_pd, team_id)[0]
        totalOppPf = self.season_totals('LPF', 'WPF', gamesWon, year_data_pd, team_id)[0]

        # Finding number of to in season
        totalTo = self.season_totals('WTO', 'LTO', gamesWon, year_data_pd, team_id)[0]
        totalOppTo = self.season_totals('LTO', 'WTO', gamesWon, year_data_pd, team_id)[0]

        # Finding number of ast in season
        totalAst = self.season_totals('WAst', 'LAst', gamesWon, year_data_pd, team_id)[0]
        totalOppAst = self.season_totals('LAst', 'WAst', gamesWon, year_data_pd, team_id)[0]

        # Finding number of Stl in season
        totalStl = self.season_totals('WStl', 'LStl', gamesWon, year_data_pd, team_id)[0]
        totalOppStl = self.season_totals('LStl', 'WStl', gamesWon, year_data_pd, team_id)[0]

        totalDrEff = totalDr / (totalOppFga + totalOppFga3 - totalOppFGM - totalOppFgm3)
        totalOrEff = totalOr / (totalFga + totalFga3 - totalFgm - totalFgm3)

        totalBlkEff = totalBlk / (totalOppFga + totalOppFGM)
        totalOppBlkEff = totalOppBlk / (totalFga + totalFgm)

        stlPerDr = totalStl / totalDr
        oppStlPerDr = totalOppStl / totalOppDr

        # Finding number of points per game allowed
        totalPointsAllowed = gamesWon['LScore'].sum()
        totalPointsAllowed += gamesLost['WScore'].sum()
        try:
            stats_SOS_pd = pd.read_csv(f'files/MMStats/{self.gender}/MMStats_{str(year)}.csv')
            stats_SOS_pd = stats_SOS_pd.rename(columns={"school_name": "School"})
            stats_SOS_pd = self.handleDifferentCSV(stats_SOS_pd)
            ratings_pd = pd.read_csv(f'files/RatingStats/{self.gender}/RatingStats_{str(year)}.csv')
            ratings_pd = ratings_pd.rename(columns={"school_name": "School", "Unnamed: 0": "Rk"})
            ratings_pd["Rk"] = ratings_pd["Rk"] + 1
            ratings_pd = self.handleDifferentCSV(ratings_pd)
            name = self.getTeamName(team_id)
            team = stats_SOS_pd[stats_SOS_pd['School'] == name]
            team_rating = ratings_pd[ratings_pd['School'] == name]
            if (len(team.index) == 0 or len(team_rating.index) == 0):  # Can't find the team
                sos = 0
                srs = 0
            else:
                sos = team['sos'].values[0]
                srs = team['srs'].values[0]
            tournamentSeed = math.floor(ratings_pd[ratings_pd["School"] == name]["Rk"] / 4)
        except:
            # logging.info(f"There is no MM file for {year}")
            sos = 0
            srs = 0
            tournamentSeed = 91

        try:
            ord_rank =         list(self.ordinals[(self.ordinals['Season'] == year) & (self.ordinals['TeamID'] == team_id)]['OrdinalRank'])[-1]
        except:
            ord_rank = 375
        # Finding tournament seed for that year
        tourneyYear = self.tourney_seeds_pd[self.tourney_seeds_pd['Season'] == year]
        seed = tourneyYear[tourneyYear['TeamID'] == team_id]
        if len(seed.index) != 0:
            seed = seed.values[0][1]
            tournamentSeed = int(seed[1:3])

        # logging.debug(f"TourneySeed is {tournamentSeed}")
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
            avgPoss = 0
            totalPointsPerPoss = 0
            totalOppPointsPerPoss = 0
            totalfgmPerPoss = 0
            totalOppFgmPerPoss = 0
            totalfgaPerPoss = 0
            totalOppFgaPerPoss = 0
            totalfgm3PerPoss = 0
            totalOppFgm3PerPoss = 0
            totalfga3PerPoss = 0
            totalOppFga3PerPoss = 0
            totalftmPerPoss = 0
            totalOppFtmPerPoss = 0
            totalftaPerPoss = 0
            totalOppFtaPerPoss = 0
            totalorPerPoss = 0
            totalOppOrPerPoss = 0
            totaldrPerPoss = 0
            totalOppDrPerPoss = 0
            totalastPerPoss = 0
            totalOppAstPerPoss = 0
            totaltoPerPoss = 0
            totalOppToPerPoss = 0
            totalstlPerPoss = 0
            totalOppStlPerPoss = 0
            totalblkPerPoss = 0
            totalOppBlkPerPoss = 0
            totalpfPerPoss = 0
            totalOppPfPerPoss = 0
            totalDrEff = 0
            totalOrEff = 0
            totalBlkEff = 0
            totalOppBlkEff = 0
            stlPerDr = 0
            oppStlPerDr = 0
        else:
            avgPointsScored = totalPointsScored / numGames
            avgPointsAllowed = totalPointsAllowed / numGames
            avg3sMade = totalFgm3 / numGames
            avgTurnovers = totalTo / numGames
            avgAssists = totalAst / numGames
            avgRebounds = totalReb / numGames
            avgSteals = totalStl / numGames
            avgPoss = totalPoss / numGames
            totalPointsPerPoss = totalPointsScored / totalPoss
            totalOppPointsPerPoss = totalPointsAllowed / totalPoss
            totalfgmPerPoss = totalFgm / totalPoss
            totalOppFgmPerPoss = totalOppFGM / totalPoss
            totalfgaPerPoss = totalFga / totalPoss
            totalOppFgaPerPoss = totalOppFga / totalPoss
            totalfgm3PerPoss = totalFgm3 / totalPoss
            totalOppFgm3PerPoss = totalOppFgm3 / totalPoss
            totalfga3PerPoss = totalFga3 / totalPoss
            totalOppFga3PerPoss = totalOppFga3 / totalPoss
            totalftmPerPoss = totalFtm / totalPoss
            totalOppFtmPerPoss = totalOppFtm / totalPoss
            totalftaPerPoss = totalFta / totalPoss
            totalOppFtaPerPoss = totalOppFta / totalPoss
            totalorPerPoss = totalOr / totalPoss
            totalOppOrPerPoss = totalOppOr / totalPoss
            totaldrPerPoss = totalDr / totalPoss
            totalOppDrPerPoss = totalOppDr / totalPoss
            totalastPerPoss = totalAst / totalPoss
            totalOppAstPerPoss = totalOppAst / totalPoss
            totaltoPerPoss = totalTo / totalPoss
            totalOppToPerPoss = totalOppTo / totalPoss
            totalstlPerPoss = totalStl / totalPoss
            totalOppStlPerPoss = totalOppStl / totalPoss
            totalblkPerPoss = totalBlk / totalPoss
            totalOppBlkPerPoss = totalOppBlk / totalPoss
            totalpfPerPoss = totalPf / totalPoss
            totalOppPfPerPoss = totalOppPf / totalPoss

        full_feature_pool = [numWins, totalPointsScored, avgPointsScored, totalPointsAllowed, avgPointsAllowed,
                             totalFgm3,
                             avg3sMade, totalTo, avgTurnovers, totalAst, avgAssists, totalReb, avgRebounds, totalStl,
                             avgSteals, totalFgm, totalfgmPerPoss, totalFga, totalfgaPerPoss, totalFgm3,
                             totalfgm3PerPoss,
                             totalFga3, totalfga3PerPoss, totalFtm, totalftmPerPoss, totalFta, totalftaPerPoss, totalOr,
                             totalorPerPoss, totalDr, totaldrPerPoss, totalastPerPoss, totaltoPerPoss, totalstlPerPoss,
                             totalBlk, totalblkPerPoss, totalPf, totalpfPerPoss, self.checkPower6Conference(team_id),
                             tournamentSeed, self.getTourneyAppearances(team_id), sos, srs]
        # full_feature_pool is trimmed to only include per possession stats
        full_feature_pool = [numWins, avgPoss, totalPointsPerPoss, totalOppPointsPerPoss, totalfgmPerPoss,
                             totalOppFgmPerPoss, totalfgaPerPoss, totalOppFgaPerPoss, totalfgm3PerPoss,
                             totalOppFgm3PerPoss, totalfga3PerPoss, totalOppFga3PerPoss, totalftmPerPoss,
                             totalOppFtmPerPoss, totalftaPerPoss, totalOppFtaPerPoss, totalorPerPoss, totalOppOrPerPoss,
                             totaldrPerPoss, totalOppDrPerPoss, totalastPerPoss, totalOppAstPerPoss, totaltoPerPoss,
                             totalOppToPerPoss, totalstlPerPoss, totalOppStlPerPoss, totalblkPerPoss,
                             totalOppBlkPerPoss,
                             totalpfPerPoss, totalOppPfPerPoss, totalDrEff, totalOrEff, totalBlkEff, totalOppBlkEff,
                             stlPerDr, oppStlPerDr, self.checkPower6Conference(team_id), tournamentSeed, ord_rank,
                             self.getTourneyAppearances(team_id), sos, srs]
        # Removed tournament seeding
        # full_feature_pool = [numWins, totalfgmPerPoss, totalfgaPerPoss, totalfgm3PerPoss, totalfga3PerPoss,
        #                      totalftmPerPoss,
        #                      totalftaPerPoss, totalorPerPoss, totaldrPerPoss, totalastPerPoss, totaltoPerPoss,
        #                      totalstlPerPoss, totalblkPerPoss, totalpfPerPoss, self.checkPower6Conference(team_id),
        #                      self.getTourneyAppearances(team_id), sos, srs]

        if cleaned_features == True:
            # This is after RFE on training data normalized with the mean
            features = [numWins, totalPointsScored, avgPointsScored, totalPointsAllowed, avgPointsAllowed, totalAst,
                        totalReb, totalFgm, totalfgmPerPoss, totalFtm, totalDr, totaldrPerPoss,
                        self.checkPower6Conference(team_id), tournamentSeed, self.getTourneyAppearances(team_id), sos,
                        srs]
        else:
            features = full_feature_pool

        float_features = [float(x) for x in features]
        return float_features

    # TODO this will need to be for 2023
    # def get2022Data(self,team_id: int, cleaned_features=True) -> List[float]:
    #     """Gets the full season data for a given team in 2022
    #
    #     This function looks at a team in 2022 and returns a list of statistics which becomes the team's vector for
    #     that season. This list of statistics can be edited and there are multiple iterations commented out in the code which
    #     have been used at various stages of testing. The list of features needs to match the list from getSeasonData.
    #
    #     Args:
    #         team_id (int): Integer id of an NCAA team
    #     Returns:
    #         List[float]
    #     """
    #
    #     # Finding number of points per game
    #     team_data = self.this_year_pd[self.this_year_pd.iloc[:, 0] == team_id]
    #     numWins = team_data.iloc[:, 3]
    #     avgPointsScored = team_data.iloc[:, 14]
    #     avgPointsAllowed = team_data.iloc[:, 15]
    #     avg3sMade = team_data.iloc[:, 20]
    #     avgAssists = team_data.iloc[:, 28]
    #     avgTurnovers = team_data.iloc[:, 31]
    #     tournamentSeed = team_data.iloc[:, 33]
    #     sos = team_data.iloc[:, 7]
    #     srs = team_data.iloc[:, 6]
    #     avgRebounds = team_data.iloc[:, 27]
    #     avgSteals = team_data.iloc[:, 29]
    #     reg_conf_champ = team_data.iloc[:, 34]
    #     conf_tourn_champ = team_data.iloc[:, 35]
    #     totalPoss = team_data.iloc[:, 36]
    #     totalfgmPerPoss = team_data.iloc[:, 38]
    #     totalfgaPerPoss = team_data.iloc[:, 39]
    #     totalfgm3PerPoss = team_data.iloc[:, 40]
    #     totalfga3PerPoss = team_data.iloc[:, 41]
    #     totalftmPerPoss = team_data.iloc[:, 42]
    #     totalftaPerPoss = team_data.iloc[:, 43]
    #     totalorPerPoss = team_data.iloc[:, 44]
    #     totaldrPerPoss = team_data.iloc[:, 45]
    #     totalastPerPoss = team_data.iloc[:, 46]
    #     totaltoPerPoss = team_data.iloc[:, 47]
    #     totalstlPerPoss = team_data.iloc[:, 48]
    #     totalblkPerPoss = team_data.iloc[:, 49]
    #     totalpfPerPoss = team_data.iloc[:, 50]
    #     #the vector needs to match the vector in getSeasonData
    #     # features = [float(numWins.iloc[0]), float(avgPointsScored), float(avgPointsAllowed), float(checkPower6Conference(team_id)),
    #     #         float(avg3sMade), float(avgAssists), float(avgTurnovers),
    #     #         float(reg_conf_champ), float(conf_tourn_champ), float(tournamentSeed),
    #     #         float(avgRebounds), float(avgSteals), float(getTourneyAppearances(team_id)),
    #     #         float(getNumChampionships(team_id)), totalPoss.iloc[0],
    #     #         totalfgmPerPoss.iloc[0], totalfgaPerPoss.iloc[0], totalfgm3PerPoss.iloc[0], totalfga3PerPoss.iloc[0],
    #     #         totalftmPerPoss.iloc[0], totalftaPerPoss.iloc[0],
    #     #         totalorPerPoss.iloc[0], totaldrPerPoss.iloc[0], totalastPerPoss.iloc[0], totaltoPerPoss.iloc[0],
    #     #         totalstlPerPoss.iloc[0], totalblkPerPoss.iloc[0], totalpfPerPoss.iloc[0]]
    #
    #     full_feature_pool = [numWins, totalPointsScored, avgPointsScored, totalPointsAllowed, avgPointsAllowed, totalFgm3,
    #                          avg3sMade, totalTo, avgTurnovers, totalAst, avgAssists, totalReb, avgRebounds, totalStl,
    #                          avgSteals, totalFgm, totalfgmPerPoss, totalFga, totalfgaPerPoss, totalFgm3, totalfgm3PerPoss,
    #                          totalFga3, totalfga3PerPoss, totalFtm, totalftmPerPoss, totalFta, totalftaPerPoss, totalOr,
    #                          totalorPerPoss, totalDr, totaldrPerPoss, totalastPerPoss, totaltoPerPoss, totalstlPerPoss,
    #                          totalBlk, totalblkPerPoss, totalPf, totalpfPerPoss, checkPower6Conference(team_id),
    #                          tournamentSeed, self.getTourneyAppearances(team_id), sos, srs]
    #
    #     if cleaned_features == True:
    #         # This is after RFE on training data normalized with the mean
    #         features = [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id), avgAssists,
    #                     avgTurnovers,
    #                     tournamentSeed, getTourneyAppearances(team_id), totalfgmPerPoss, totalftmPerPoss, totalftaPerPoss,
    #                     totaldrPerPoss, totalastPerPoss]
    #     else:
    #         features = full_feature_pool
    #
    #     float_features = [float(x) for x in features]
    #     return float_features
    #

    def compareTwoTeams(self, id_1: int, id_2: int, year: int) -> List[float]:
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
        team_1 = self.getSeasonData(id_1, year)
        team_2 = self.getSeasonData(id_2, year)
        diff = [a - b for a, b in zip(team_1, team_2)]
        return diff

    def createSeasonDict(self, year: int) -> Dict[int, List[float]]:
        """Creates a dictionary of team vectors for all teams in a given year.

        This function looks at all teams in a given year and calculates their team vector. It then creates a dictionary
        where the team_id is the key and the vector is the value.

        Args:
            year (int): Full year
        Returns:
            Dict[int,List[float]]
        """

        seasonDictionary = collections.defaultdict(list)
        for team in self.teamList:
            team_id = self.teams_pd[self.teams_pd['TeamName'] == team].values[0][0]
            team_vector = self.getSeasonData(team_id, year)
            seasonDictionary[team_id] = team_vector
        return seasonDictionary

    def getHomeStat(self, row: pd.Series) -> int:
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

    def get_season_for_tourney_teams(self, season, tourney):
        teams = list(tourney['WTeamID'])
        teams.extend(list(tourney['LTeamID']))
        teams = list(set(teams))
        season_trimmed = season[(season['WTeamID'].isin(teams)) & (season['LTeamID'].isin(teams))]
        return season_trimmed

    def createTrainingSet(self, years: range) -> Tuple[np.array, np.array]:
        """Creates a training set and a training target set

        This function iterates through a range of years and creates a training set of matchup vectors as well as a target
        vector indicating if the first named team in the matchup won or lost.

        Args:
            years (range): A range object of years to include in the training set
        Returns:
            Tuple[np.array,np.array]
        """

        totalNumGames = 0
        # loop through years
        for year in years:
            season = self.filtered_regular[self.filtered_regular['Season'] == year]
            tourney = self.tourney_compact_pd[self.tourney_compact_pd['Season'] == year]
            # season_trimmed = self.get_season_for_tourney_teams(season, tourney)

            totalNumGames += len(season.index)
            totalNumGames += len(tourney.index)
        numFeatures = len(
            self.getSeasonData(1181, 2012))  # Just choosing a random team and seeing the dimensionality of the vector
        x_Train = np.zeros((totalNumGames, numFeatures + 1))
        y_Train = np.zeros(totalNumGames)
        indexCounter = 0
        for year in years:
            team_vectors = self.createSeasonDict(year)
            season = self.filtered_regular[self.filtered_regular['Season'] == year]
            tourney = self.tourney_compact_pd[self.tourney_compact_pd['Season'] == year]
            # season_trimmed = self.get_season_for_tourney_teams(season, tourney)
            numGamesInSeason = len(season.index)
            numGamesInSeason += len(tourney.index)
            xTrainSeason = np.zeros((numGamesInSeason, numFeatures + 1))
            yTrainSeason = np.zeros((numGamesInSeason))
            counter = 0
            for index, row in season.iterrows():
                w_team = row['WTeamID']
                w_vector = team_vectors[w_team]
                l_team = row['LTeamID']
                l_vector = team_vectors[l_team]
                diff = [a - b for a, b in zip(w_vector, l_vector)]
                home = self.getHomeStat(row['WLoc'])
                if (counter % 2 == 0):
                    diff.append(home)
                    xTrainSeason[counter] = diff
                    yTrainSeason[counter] = 1
                else:
                    diff.append(-home)
                    xTrainSeason[counter] = [-p for p in diff]
                    yTrainSeason[counter] = 0
                counter += 1
            for index, row in tourney.iterrows():
                w_team = row['WTeamID']
                w_vector = team_vectors[w_team]
                l_team = row['LTeamID']
                l_vector = team_vectors[l_team]
                diff = [a - b for a, b in zip(w_vector, l_vector)]
                home = 0  # All tournament games are neutral
                if (counter % 2 == 0):
                    diff.append(home)
                    xTrainSeason[counter] = diff
                    yTrainSeason[counter] = 1
                else:
                    diff.append(-home)
                    xTrainSeason[counter] = [-p for p in diff]
                    yTrainSeason[counter] = 0
                counter += 1
            x_Train[indexCounter:numGamesInSeason + indexCounter] = xTrainSeason
            y_Train[indexCounter:numGamesInSeason + indexCounter] = yTrainSeason
            indexCounter += numGamesInSeason
        return x_Train, y_Train

    # TODO normalize inputs
    def normalizeInput(self, arr: List[float]) -> List[float]:
        """Normalizes values in a vector

        This function scales the values in the vectors between the max and the min values in the data.

        Args:
            arr (List[float]): A feature vector
        Returns:
            List[float]
        """
        arr_new = arr.copy()
        self.training_min = np.min(self.xTrain, axis=0)
        self.training_max = np.max(self.xTrain, axis=0)
        for j in range(arr.shape[1]):
            minVal = min(arr[:, j])
            maxVal = max(arr[:, j])
            arr_new[:, j] = (arr_new[:, j] - minVal) / (maxVal - minVal)
        return arr_new

    # alternative:
    def normalize(self, X: List[float]) -> List[float]:
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
        training_mean = np.mean(X, axis=0)
        training_std = np.std(X, axis=0)
        return (x_new - np.mean(x_new, axis=0)) / np.std(x_new, axis=0)

    def showDependency(self, predictions, test, stat, my_categories):
        """Plots the actual values vs predictions for a given stat

        Args:
            predictions (List): The list of predictions generated by the model
            test ():
            stat ():
            my_categories():
        Returns:
            None
        """

        difference = test[:, my_categories.index(stat)]
        plt.scatter(difference, predictions)
        plt.ylabel('Probability of Team 1 Win')
        plt.xlabel(stat + ' Difference (Team 1 - Team 2)')
        plt.show()

    def showFeatureImportance(self, my_categories):
        """Plots feature importance from model

        Args:
            my_categories (List[str]): A list of features to check the importance of
        Returns:
            None
        """

        fx_imp = pd.Series(self.model.feature_importances_, index=my_categories)
        fx_imp /= fx_imp.max()
        fx_imp.sort_values(ascending=True)
        fx_imp.plot(kind='barh')

    def predictGame(self, team_1_vector: List[float], team_2_vector: List[float], home: int) -> float:
        """Runs a matchup vector through the trained model and outputs a probability that the first team wins

        Args:
            team_1_vector (List[float]): The team vector of the first team
            team_2_vector (List[float]): The team vector of the second team
            home (int): If the first team is home, this is 1. Away is -1, and neutral is 0
        Returns:
            float
        """

        diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
        diff.append(float(home))
        diff = pd.DataFrame(diff).T
        if self.rfecv is not None:
            diff = diff[self.rfecv.get_support(indices=True)]
            diff = (diff - self.training_min[self.rfecv.get_support(indices=True)]) / (
                        self.training_max[self.rfecv.get_support(indices=True)] - self.training_min[
                    self.rfecv.get_support(indices=True)])
        else:
            diff = (diff - self.training_min) / (self.training_max - self.training_min)

        probability_team_1 = (self.clsf.model.predict_proba([diff.loc[0]])[0][1])

        diff_rev = [a - b for a, b in zip(team_2_vector, team_1_vector)]
        home_flip = home
        if home == 1:
            home_flip = -1
        elif home == -1:
            home_flip = 1
        diff_rev.append(float(home_flip))
        diff_rev = pd.DataFrame(diff_rev).T
        if self.rfecv is not None:
            diff_rev = diff_rev[self.rfecv.get_support(indices=True)]
            diff_rev = (diff_rev - self.training_min[self.rfecv.get_support(indices=True)]) / (
                    self.training_max[self.rfecv.get_support(indices=True)] - self.training_min[
                self.rfecv.get_support(indices=True)])
        else:
            diff_rev = (diff_rev - self.training_min) / (self.training_max - self.training_min)

        probability_team_2 = (self.clsf.model.predict_proba([diff_rev.loc[0]])[0][1])
        avg_prob = (probability_team_1 + 1 - probability_team_2) / 2
        return avg_prob
        # return model.predict_proba([diff])

    # TODO experiment with the range of years which leads to the best results
    def get_x_and_y(self, load_model: bool, start_year: int, end_year: int) -> Tuple[np.array, np.array]:
        """Creates training data from all data between two years.

        Args:
            load_model (bool): If True, load prebuilt data
            start_year (int): first year in dataset
            end_year (int): last year in dataset
        Returns:
            Tuple[np.array,np.array]
        """
        if load_model == False:
            years_to_train = range(start_year, end_year)
            logging.info("Creating training set")
            self.xTrain, self.yTrain = self.createTrainingSet(years_to_train)
            logging.info("Training set created")
            np.save(f'{self.gender}xTrain_3', self.xTrain)
            np.save(f'{self.gender}yTrain_3', self.yTrain)
        else:
            self.xTrain = np.load(f'{self.gender}xTrain.npy')
            self.yTrain = np.load(f'{self.gender}yTrain.npy')
        return self.xTrain, self.yTrain

    def build_model(self, gender: str, prebuilt_data: bool, first_year: int, last_year: int, gridsearch_flag: str,
                    rfecv_flag: bool,
                    analyze_flag: bool) -> classifier.Classifier:
        """
        Full program. Reads data, builds Random Forests model.

        Flags allow for automated parameter tuning using gridsearch. Feature selection and feature selection
        Args:
            gender (str): men or women
            load_model (bool): If True, load prebuilt data
            start_year (int): first year in dataset
            end_year (int): last year in dataset
        Returns:
            Tuple[np.array,np.array]
        """
        self.gender = gender
        self.read_data(gender)

        # create lists of teams in major conferences
        self.listACCteams = ['North Carolina', 'Virginia', 'Florida St', 'Louisville', 'Notre Dame', 'Syracuse', 'Duke',
                             'Virginia Tech', 'Georgia Tech', 'Miami', 'Wake Forest', 'Clemson', 'NC State',
                             'Boston College', 'Pittsburgh']
        self.listPac12teams = ['Arizona', 'Oregon', 'UCLA', 'California', 'USC', 'Utah', 'Washington St', 'Stanford',
                               'Arizona St', 'Colorado', 'Washington', 'Oregon St']
        self.listSECteams = ['Kentucky', 'South Carolina', 'Florida', 'Arkansas', 'Alabama', 'Tennessee',
                             'Mississippi St', 'Georgia', 'Ole Miss', 'Vanderbilt', 'Auburn', 'Texas A&M', 'LSU',
                             'Missouri']
        self.listBig10teams = ['Maryland', 'Wisconsin', 'Purdue', 'Northwestern', 'Michigan St', 'Indiana', 'Iowa',
                               'Michigan', 'Penn St', 'Nebraska', 'Minnesota', 'Illinois', 'Ohio St', 'Rutgers']
        self.listBig12teams = ['Kansas', 'Baylor', 'West Virginia', 'Iowa St', 'TCU', 'Kansas St', 'Texas Tech',
                               'Oklahoma St', 'Texas', 'Oklahoma']
        self.listBigEastteams = ['Butler', 'Creighton', 'DePaul', 'Georgetown', 'Marquette', 'Providence', 'Seton Hall',
                                 'St John\'s', 'Villanova', 'Xavier']

        # TODO this is unused
        # getListForURL(teamList)

        if gender == "men":
            # test the functions to this point
            print("The vector for teamID 1103 in 2022 is ", self.getSeasonData(1103, 2022))

            # get kentucky vector from 2021
            kentucky_id = self.teams_pd[self.teams_pd['TeamName'] == 'Kentucky'].values[0][0]
            print("The vector for Kentucky in 2021 is ", self.getSeasonData(kentucky_id, 2021))

            # test comparison of two teams in 2024
            kansas_id = self.teams_pd[self.teams_pd['TeamName'] == 'Kansas'].values[0][0]
            print("The vector for teamIDs 1234 and 1242 in 2024 is ", self.compareTwoTeams(1234, 1242, 2023))
        else:
            # test the functions to this point
            print("The vector for teamID 3103 in 2024 is ", self.getSeasonData(3103, 2024))

            # get kentucky vector from 2021
            kentucky_id = self.teams_pd[self.teams_pd['TeamName'] == 'Kentucky'].values[0][0]
            print("The vector for Kentucky in 2021 is ", self.getSeasonData(kentucky_id, 2021))

            # test comparison of two teams in 2022
            kansas_id = self.teams_pd[self.teams_pd['TeamName'] == 'Kansas'].values[0][0]
            print("The vector for teamIDs 3234 and 3242 in 2024 is ", self.compareTwoTeams(3234, 3242, 2024))

        training_data = self.get_x_and_y(prebuilt_data, first_year, last_year)
        self.xTrain = training_data[0]
        self.yTrain = training_data[1]
        self.xTrainNorm = self.normalizeInput(self.xTrain)
        print("xTrain shape: ", self.xTrain.shape, "\nyTrain shape: ", self.yTrain.shape)

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
        model = RandomForestClassifier(n_jobs=-1, bootstrap=False, max_depth=None, max_features='auto',
                                       min_samples_leaf=20,
                                       min_samples_split=9, n_estimators=150)
        # model = KNeighborsClassifier(n_neighbors=39)
        # neuralNetwork(10)
        # model = VotingClassifier(estimators=[('GBR', model1), ('BR', model2), ('KNN', model3)], voting='soft')
        # model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=0.1)
        # model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
        # model = linear_model.Ridge(alpha = 0.5)
        model = xgb.XGBClassifier(verbosity=1, n_jobs=-1)
        # TODO utilize the categories or delete them

        self.clsf = classifier.Classifier(model, self.xTrainNorm, self.yTrain,
                                          ['numWins', 'avgPoss', 'totalPointsPerPoss', 'totalOppPointsPerPoss', 'totalfgmPerPoss',
                                           'totalOppFgmPerPoss', 'totalfgaPerPoss', 'totalOppFgaPerPoss', 'totalfgm3PerPoss',
                                           'totalOppFgm3PerPoss', 'totalfga3PerPoss', 'totalOppFga3PerPoss', 'totalftmPerPoss',
                                           'totalOppFtmPerPoss', 'totalftaPerPoss', 'totalOppFtaPerPoss', 'totalorPerPoss',
                                           'totalOppOrPerPoss',
                                           'totaldrPerPoss', 'totalOppDrPerPoss', 'totalastPerPoss', 'totalOppAstPerPoss',
                                           'totaltoPerPoss',
                                           'totalOppToPerPoss', 'totalstlPerPoss', 'totalOppStlPerPoss', 'totalblkPerPoss',
                                           'totalOppBlkPerPoss',
                                           'totalpfPerPoss', 'totalOppPfPerPoss', 'totalDrEff', 'totalOrEff', 'totalBlkEff',
                                           'totalOppBlkEff',
                                           'stlPerDr', 'oppStlPerDr', 'checkPower6Conference', 'tournamentSeed',
                                           'ord_rank', 'tourney_appearancs', 'sos', 'srs'])

        # paramgrid = {'bootstrap': [True, False],
        #              'max_depth': [10, None],
        #              'max_features': ['auto'],
        #              'min_samples_leaf': [10, 20],
        #              'min_samples_split': [5, 9],
        #              'n_estimators': [50, 150]}
        if gridsearch_flag == 'grid':
            self.clsf.gridSearch(paramgrid={'eta': [.3, .5, .7],
                                            'max_depth': [3, 6],
                                            'subsample': [.5, 1]})
        if gridsearch_flag == 'bayes':
            self.clsf.bayesian_opt()
            self.clsf.model['max_depth'] = int(self.clsf.model['max_depth'])

        if rfecv_flag == True:
            self.rfecv = self.clsf.RFECVSelect()

        if analyze_flag == True:
            accuracy = self.clsf.analyze_model()

        self.clsf.model.fit(self.clsf.xTrain, self.clsf.yTrain)

        # This can be used to predict 2022 games
        team1_name = "Creighton"
        team2_name = "San Diego St"
        team1_vector = self.getSeasonData(self.teams_pd[self.teams_pd['TeamName'] == team1_name].values[0][0], 2023)
        team2_vector = self.getSeasonData(self.teams_pd[self.teams_pd['TeamName'] == team2_name].values[0][0], 2023)

        print('Probability that ' + team1_name + ' wins:', self.predictGame(team1_vector, team2_vector, 0))
        try:
            print(self.rfecv.cv_results_['mean_test_score'].mean())
            print(self.rfecv.cv_results_['std_test_score'].mean())
        except:
            logging.error("Tried to reference rfecv with flag set to False")

        # pickle.dump(self, open("2023_march_madness.pickle", 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{self.gender}_noseed_xgb_2023_march_madness_3.dill', 'wb') as file:
            pickle.dump(self, file)
        return self.clsf

    def submission(self, file_out="files/2023MarchMadnessKaggleLMR.csv") -> None:
        submission = pd.read_csv("files/SampleSubmission2024.csv")
        submission
        logging.info("Creating submission csv")
        preds = []
        for i in range(130683):
            if submission.iloc[i][0][5] == "1":
                vector1 = self.getSeasonData(int(submission.iloc[i][0][5:9]), 2024)
                vector2 = self.getSeasonData(int(submission.iloc[i][0][10:14]), 2024)
                pred = self.predictGame(vector1, vector2, 0)
                preds.append(pred)
        submission["Pred"] = preds
        #
        #
        #
        submission.tail()

        submission.to_csv(file_out)
        logging.info("Submission csv created successfully")


if __name__ == "__main__":

    def load_model():
        try:
            file = open("combined_2023_march_madness.dill", "rb")
            run = dill.load(file)
            file.close()
            return run
        except:
            logging.error("No saved model. Building model.")
            return


    def createSeasonDict():
        season_dict = {}
        for i in range(len(men_data_obj.teams_pd)):
            id = men_data_obj.teams_pd["TeamID"][i]
            season_dict[id] = men_data_obj.getSeasonData(id, 2023)
            print(i)

        for i in range(len(women_data_obj.teams_pd)):
            id = women_data_obj.teams_pd["TeamID"][i]
            season_dict[id] = women_data_obj.getSeasonData(id, 2023)
            print(i)
        return season_dict


    def submission1(string):
        vector1 = season_dict[int(string[5:9])]
        vector2 = season_dict[int(string[10:14])]
        prediction = men_data_obj.predictGame(vector1, vector2, 0)
        return prediction


    def submission2(string):
        vector1 = season_dict[int(string[5:9])]
        vector2 = season_dict[int(string[10:14])]
        prediction = women_data_obj.predictGame(vector1, vector2, 0)
        return prediction


    def createSubmission(mens_model, womens_model, file_out="files/2023NoSeedMarchMadnessKaggleLMR.csv") -> None:
        submission = pd.read_csv("files/SampleSubmission2023.csv")
        sub_men = submission[:65703]
        sub_women = submission[65703:]
        logging.info("Creating submission csv")
        sub_men["pred2"] = sub_men["ID"].apply(submission1)
        sub_women["pred2"] = sub_women["ID"].apply(submission2)
        new_preds = sub_men["pred2"].append(sub_women["pred2"])
        submission["pred"] = new_preds
        submission.drop(["Pred"], axis=1)
        submission = submission.copy().drop(["Pred"], axis=1)
        submission = submission.copy().rename(columns={"pred": "Pred"})
        submission.to_csv("files/2023NoSeedMarchMadnessKaggleLMR.csv", index=False)

        submission.to_csv(file_out)
        logging.info("Submission csv created successfully")


    # def xgBoost():
    #     import xgboost as xgb
    #     xgb_model = xgb.XGBClassifier(verbosity=2)
    #     xgb_model.fit(X_train, y_train)

    men_data_obj = Data()
    women_data_obj = Data()
    men_classifier = men_data_obj.build_model("men", False, 2003, 2023, 'bayes', False, True)
    # women_classifier = women_data_obj.build_model("women", False, 2010, 2024, False, False, True)
    # season_dict = createSeasonDict()
    # createSubmission(men_data_obj, women_data_obj)
