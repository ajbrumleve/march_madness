import collections
import math
import os
import re
import csv
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, brier_score_loss
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns

from AutoBlockCV import AutoBlockCV


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
        self.accuracy = {}
        self.season_dict_file = '2025_saved_data/season_dicts.pkl'

    def get_key_players(self,team_id,season):
        file = f'2025_saved_data/{self.gender}PlayerQuality2.csv'
        key_player_df = pd.read_csv(file)
        filtered_df = key_player_df[(key_player_df['Season']==season) & (key_player_df['TeamID']==team_id)].reset_index()
        try:
            quality_g_count = filtered_df.iloc[0]['Quality_G_Count']
            quality_f_count = filtered_df.iloc[0]['Quality_F_Count']
        except:
            quality_g_count = 0
            quality_f_count = 0
        return quality_g_count, quality_f_count

    def read_data(self, gender):
        def filter_by_season(row, tourney_teams):
            return (row['WTeamID'] in tourney_teams) and (row['LTeamID'] in tourney_teams)

        # read in data
        if gender == "men":
            self.gender = "M"
            self.season_dict_file = f'2025_saved_data/{self.gender}_season_dicts.pkl'
            # read compact data in as DataFrame
            self.reg_season_compact_pd = pd.read_csv(r'files/MRegularSeasonCompactResults.csv')
            print("Regular Season Compact\n\n", self.reg_season_compact_pd.head(), )
            self.team_conference_pd = pd.read_csv(r'files/MTeamConferences.csv')
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
            # self.get_wins_above_seed()
            print("These are a sample of the tourney seeds\n\n", self.tourney_seeds_pd.head())
            self.read_ordinals()
            merged_df = pd.merge(self.reg_season_detailed_pd, self.ordinals, how='left',
                                 left_on=['Season', 'DayNum', 'WTeamID'],
                                 right_on=['Season', 'RankingDayNum', 'TeamID'])
            merged_df = pd.merge(merged_df, self.ordinals, how='left', left_on=['Season', 'DayNum', 'LTeamID'],
                                 right_on=['Season', 'RankingDayNum', 'TeamID'], suffixes=('_W', '_L'))
            merged_df['OrdinalRank_W'] = merged_df.groupby(['Season','WTeamID'])['OrdinalRank_W'].fillna(method='ffill')
            merged_df['OrdinalRank_L'] = merged_df.groupby(['Season','LTeamID'])['OrdinalRank_L'].fillna(method='ffill')

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
            self.gender = "W"
            # read compact data in as DataFrame
            self.season_dict_file = f'2025_saved_data/{self.gender}_season_dicts.pkl'
            self.reg_season_compact_pd = pd.read_csv(r'files/WRegularSeasonCompactResults.csv')
            print("Regular Season Compact\n\n", self.reg_season_compact_pd.head(), )
            self.team_conference_pd = pd.read_csv(r'files/WTeamConferences.csv')
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
            # self.get_wins_above_seed()
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
        # ranking_days = sorted(df['RankingDayNum'].unique().tolist() + [129, 130])
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

    def checkPower6Conference(self, team_id: int, year: int) -> int:
        """Checks if a team is in a power conference

        This checks if a team is in oneof the top 6 larger conferenes in the NCAA it returns 1 if so and 0 if not

        Args:
            team_id (int): The integer id of a given team
        Returns:
            int
        """
        try:
            conference = self.team_conference_pd[self.team_conference_pd['TeamID'] == team_id]['ConfAbbrev'].iloc[-1]
        except:
            return 0
        if conference in ['acc', 'big_ten', 'big_twelve', 'sec','pac_twelve','big_east']:
            return 1
        else:
            return 0
        # try:
        #     teamName = self.teams_pd.loc[self.teams_pd['TeamID'] == team_id, 'TeamName'].iloc[0]
        #     # teamName = self.teams_pd.values[team_id - 1101][1]
        # except:
        #     return 0
        # if (teamName in self.listACCteams or teamName in self.listBig10teams or teamName in self.listBig12teams
        #         or teamName in self.listSECteams or teamName in self.listPac12teams or teamName in self.listBigEastteams):
        #     return 1
        # else:
        #     return 0

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

    def season_totals(self, wstat: str, lstat: str, gamesWon: pd.DataFrame, df: pd.DataFrame, team_id: int) -> \
    Tuple[
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
        Quality_G_Count, Quality_F_Count = self.get_key_players(team_id, year)
        try:
            if self.gender == 'M':
                gender_label = 'men'
            elif self.gender == "W":
                gender_label = 'women'
            stats_SOS_pd = pd.read_csv(f'files/MMStats/{gender_label}/MMStats_{str(year)}.csv')
            stats_SOS_pd = stats_SOS_pd.rename(columns={"school_name": "School"})
            stats_SOS_pd = self.handleDifferentCSV(stats_SOS_pd)
            ratings_pd = pd.read_csv(f'files/RatingStats/{gender_label}/RatingStats_{str(year)}.csv')
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
            ord_rank = list(self.ordinals[(self.ordinals['Season'] == year) & (self.ordinals['TeamID'] == team_id)][
                                'OrdinalRank'])[-1]
        except:
            ord_rank = 375
        # Finding tournament seed for that year
        tourneyYear = self.tourney_seeds_pd[self.tourney_seeds_pd['Season'] == year]
        seed = tourneyYear[tourneyYear['TeamID'] == team_id]
        if len(seed.index) != 0:
            seed = seed.values[0][1]
            # tournamentSeed = int(seed[1:3])
            tournamentSeed = extract_number(seed)
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
                         avg3sMade, totalTo, avgTurnovers, totalAst, avgAssists, totalReb, avgRebounds,
                         totalStl,
                         avgSteals, totalFgm, totalfgmPerPoss, totalFga, totalfgaPerPoss, totalFgm3,
                         totalfgm3PerPoss,
                         totalFga3, totalfga3PerPoss, totalFtm, totalftmPerPoss, totalFta, totalftaPerPoss,
                         totalOr,
                         totalorPerPoss, totalDr, totaldrPerPoss, totalastPerPoss, totaltoPerPoss,
                         totalstlPerPoss,
                         totalBlk, totalblkPerPoss, totalPf, totalpfPerPoss,
                         self.checkPower6Conference(team_id, year),
                         tournamentSeed, self.getTourneyAppearances(team_id), sos, srs, Quality_G_Count, Quality_F_Count]

        # full_feature_pool is trimmed to only include per possession stats
        full_feature_pool = [numWins, avgPoss, totalPointsPerPoss, totalOppPointsPerPoss, totalfgmPerPoss,
                             totalOppFgmPerPoss, totalfgaPerPoss, totalOppFgaPerPoss, totalfgm3PerPoss,
                             totalOppFgm3PerPoss, totalfga3PerPoss, totalOppFga3PerPoss, totalftmPerPoss,
                             totalOppFtmPerPoss, totalftaPerPoss, totalOppFtaPerPoss, totalorPerPoss,
                             totalOppOrPerPoss,
                             totaldrPerPoss, totalOppDrPerPoss, totalastPerPoss, totalOppAstPerPoss, totaltoPerPoss,
                             totalOppToPerPoss, totalstlPerPoss, totalOppStlPerPoss, totalblkPerPoss,
                             totalOppBlkPerPoss,
                             totalpfPerPoss, totalOppPfPerPoss, totalDrEff, totalOrEff, totalBlkEff, totalOppBlkEff,
                             stlPerDr, oppStlPerDr, self.checkPower6Conference(team_id, year), tournamentSeed, ord_rank,
                             self.getTourneyAppearances(team_id), sos, srs, Quality_G_Count, Quality_F_Count]
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
                        self.checkPower6Conference(team_id, year), tournamentSeed, self.getTourneyAppearances(team_id),
                        sos,
                        srs]
        else:
            features = full_feature_pool

        float_features = [float(x) for x in features]
        return float_features

    def recursive_feature_elimination(self, X: np.ndarray, y: np.ndarray, feature_names: list, estimator=None, cv: int = 5):
        """
        Performs Recursive Feature Elimination with Cross-Validation (RFECV) and returns
        the reduced feature set and a list of dropped columns.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target array.
            feature_names (list): List of feature names.
            estimator (object, optional): Estimator to use for RFE. Defaults to LogisticRegression().
            cv (int, optional): Number of cross-validation folds. Defaults to 5.

        Returns:
            tuple: (X_new, dropped_features)
        """
        if estimator is None:
            estimator = LogisticRegression()
        scaler = StandardScaler().fit(X)
        X_Scaled = scaler.transform(X)
        # rfecv = RFECV(estimator, step=1, cv=cv, scoring='precision',min_features_to_select=10)
        # Special case for GaussianNB
        if isinstance(estimator, GaussianNB):
            rfecv = RFECV(estimator, step=1, cv=cv, scoring='accuracy', min_features_to_select=10,
                          importance_getter=lambda model: np.var(model.theta_, axis=0))
        else:
            rfecv = RFECV(estimator, step=1, cv=cv, scoring='accuracy', min_features_to_select=10)
        rfecv.fit(X_Scaled, y)

        selected_mask = rfecv.support_  # Boolean mask of selected features
        selected_features = np.array(feature_names)[rfecv.support_].tolist()
        dropped_features = np.array(feature_names)[~rfecv.support_].tolist()

        X_new = X[:, rfecv.support_]

        # Extract feature importance if available
        if hasattr(estimator, 'coef_'):
            importance_values = np.abs(rfecv.estimator_.coef_).flatten()  # Absolute values for sorting
        elif hasattr(estimator, 'feature_importances_'):
            importance_values = rfecv.estimator_.feature_importances_
        else:
            importance_values = None  # Some models do not have importance scores

        if importance_values is not None:
            feature_importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': importance_values
            }).sort_values(by='Importance', ascending=False)
        else:
            feature_importance_df = None  # Cannot compute importance

        return X_new, dropped_features, feature_importance_df, selected_mask
        # return X_new, dropped_features

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
        from tqdm import tqdm
        seasonDictionary = collections.defaultdict(list)
        for team in tqdm(self.teamList):
            team_id = self.teams_pd[self.teams_pd['TeamName'] == team].values[0][0]
            team_vector = self.getSeasonData(team_id, year)
            seasonDictionary[team_id] = team_vector
        return seasonDictionary

    def get_tourney_wins(self, team_id, year):
        team_tourney_wins = self.tourney_compact_pd[(self.tourney_compact_pd['WTeamID']==team_id) & (self.tourney_compact_pd['Season']==year)]
        return team_tourney_wins.shape[0]

    def get_expected_tourney_wins(self, seed):
        win_map = {1:4,
                   2:3,
                   3:2,
                   4:2,
                   5:1,
                   6:1,
                   7:1,
                   8:1,
                   9:0,
                   10:0,
                   11:0,
                   12:0,
                   13:0,
                   14:0,
                   15:0,
                   16:0
         }
        return win_map[seed]
    def get_wins_above_seed(self):
        df = self.tourney_seeds_pd
        df['Seed'] = df['Seed'].apply(extract_number)
        df['expected_wins'] = df['Seed'].apply(lambda seed: self.get_expected_tourney_wins(seed))
        df['wins'] = df.apply(lambda row: self.get_tourney_wins(row['TeamID'], row['Season']), axis=1)
        df['WAS'] = df['wins'] - df['expected_wins']
        self.tourney_seeds_pd = df

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

    def createWOSTrainingSet(self, years: range):
        from tqdm import tqdm
        X_list = []
        y_list = []
        for year in tqdm(years):
            tourney_teams_season = self.tourney_seeds_pd[self.tourney_seeds_pd['Season']==year]
            for index, row in tqdm(tourney_teams_season.iterrows()):
                X = self.getSeasonData(row['TeamID'], row['Season'])
                y = row['WAS']
                X_list.append(X)
                y_list.append(y)

        # Convert lists to arrays (optional, if using numpy for ML models)
        X = np.array(X_list)
        y = np.array(y_list)

        self.WAS_X = X
        self.WAS_y = y
        return X,y

    def train_WAS_model(self):
        # Initialize K-Fold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        train_rmse_scores = []  # Store training RMSE scores
        val_rmse_scores = []  # Store validation RMSE scores

        # Loop through each fold
        for train_index, test_index in kf.split(X):
            X_train, X_test = self.WAS_X[train_index], self.WAS_X[test_index]
            y_train, y_test = self.WAS_y[train_index], self.WAS_y[test_index]
            y_train_class = np.where(y_train < -1, 0, np.where(y_train > 1, 2, 1))
            y_test_class = np.where(y_test < -1, 0, np.where(y_test > 1, 2, 1))

            # Initialize XGBoost Regressor
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                alpha=.95,
                n_estimators=50,
                max_depth=3,  # Reduce tree depth to prevent overfitting
                learning_rate=0.05,  # Lower learning rate for smoother training
                subsample=0.8,  # Use only 80% of training data per tree
                colsample_bytree=0.8,  # Use only 80% of features per tree
                reg_alpha=0.1,  # L1 Regularization (lasso)
                reg_lambda=1.0,  # L2 Regularization (ridge)
                random_state=42
            )
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=100,  # Increase trees
                max_depth=4,  # Allow more complexity
                learning_rate=0.03,  # Lower LR with more trees
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,  # Reduce L1 regularization
                reg_lambda=0.8,  # Reduce L2 regularization
                random_state=42
            )
            model = XGBClassifier(
                objective="multi:softmax",
                num_class = 9,
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )
            # Train the model
            model.fit(
                X_train, y_train_class
            )

            # Make predictions
            y_train_pred = model.predict(X_train)  # Predictions on training data
      # Predictions on training data
            y_test_pred = model.predict(X_test)
            import matplotlib.pyplot as plt

            plt.scatter(y_test_class, y_test_pred, alpha=0.5)
            plt.xlabel("Actual Y")
            plt.ylabel("Predicted Y")
            plt.title("Actual vs. Predicted Values")
            plt.axline((0, 0), slope=1, color="red", linestyle="--")  # Ideal line
            plt.show()
            # Compute RMSE
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # Training RMSE
            val_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # Validation RMSE

            train_rmse_scores.append(train_rmse)
            val_rmse_scores.append(val_rmse)

        # Compute average RMSE across all folds
        avg_train_rmse = np.mean(train_rmse_scores)
        avg_val_rmse = np.mean(val_rmse_scores)

        # Print results
        print(f"Train RMSE for each fold: {train_rmse_scores}")
        print(f"Validation RMSE for each fold: {val_rmse_scores}")
        print(f"Average Train RMSE: {avg_train_rmse:.4f}")
        print(f"Average Validation RMSE: {avg_val_rmse:.4f}")

        # Overfitting Detection
        if avg_train_rmse < avg_val_rmse * 0.75:  # Adjust the threshold as needed
            print("🚨 Possible Overfitting: Training RMSE is much lower than Validation RMSE")
        elif avg_train_rmse > avg_val_rmse * 0.95:
            print("✅ Model seems well-generalized.")
        else:
            print("⚠️ Mild Overfitting: There is a gap, but it might be acceptable.")
        with open('WAS_model.mdl','wb') as outp:
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
        return model




    def train_classifiers(self, X, y, metric='precision'):
        """
        Trains multiple classifier models using K-Fold cross-validation, evaluates their performance,
        checks for overfitting, and saves the trained models.

        Parameters:
            metric (str): The metric to optimize. Options: 'precision', 'recall', 'f1', 'accuracy'.
                          Default is 'precision'.
        """
        features = ['numWins', 'avgPoss', 'totalPointsPerPoss', 'totalOppPointsPerPoss', 'totalfgmPerPoss',
                    'totalOppFgmPerPoss', 'totalfgaPerPoss', 'totalOppFgaPerPoss', 'totalfgm3PerPoss',
                    'totalOppFgm3PerPoss', 'totalfga3PerPoss', 'totalOppFga3PerPoss', 'totalftmPerPoss',
                    'totalOppFtmPerPoss', 'totalftaPerPoss', 'totalOppFtaPerPoss', 'totalorPerPoss',
                    'totalOppOrPerPoss',
                    'totaldrPerPoss', 'totalOppDrPerPoss', 'totalastPerPoss', 'totalOppAstPerPoss', 'totaltoPerPoss',
                    'totalOppToPerPoss', 'totalstlPerPoss', 'totalOppStlPerPoss', 'totalblkPerPoss',
                    'totalOppBlkPerPoss',
                    'totalpfPerPoss', 'totalOppPfPerPoss', 'totalDrEff', 'totalOrEff', 'totalBlkEff', 'totalOppBlkEff',
                    'stlPerDr', 'oppStlPerDr', 'Power6Conference', 'tournamentSeed', 'ord_rank',
                    'TourneyAppearances', 'sos', 'srs', 'Quality_G_Count', 'Quality_F_Count', 'home']
        new_X, dropped_features, importance_df_nb, selected_mask = self.recursive_feature_elimination(
            X=X, y=y,
            feature_names=features,
            estimator=XGBClassifier())
        self.main_mask = selected_mask
        # Initialize K-Fold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv = AutoBlockCV(self.tourney_start_index)
        # xTrain, yTrain, xVal, yVal = X[:-67], y[:-67], X[-67:], y[-67:]
        xTrain, yTrain = new_X,y
        # Define models with hyperparameters for Grid Search
        models = {
            "XGBoost": (XGBClassifier(objective='binary:logistic', reg_lambda=1, reg_alpha=.5, random_state=42),
                        {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1],'reg_lambda':[0,.5,1], 'reg_alpha':[0,0.5,1]}),

            # "RandomForest": (RandomForestClassifier(random_state=42),
            #                  {'n_estimators': [100, 200], 'max_depth': [10, None], 'min_samples_split': [2, 5]}),

            "LogisticRegression": (LogisticRegression(solver='liblinear', random_state=42),
                                   {'C': [0.1, .5, 1, 2, 5, 10]}),
            "Stacked":("",{}),
            "Ensemble":("",{})
        }

        results = {}
        models_dict = {}
        for model_name, (model, param_grid) in models.items():
            if model_name == "Stacked" or model_name == 'Ensemble':
                print(f"\n🚀 Training {model_name} ...\n")
            else:
                print(f"\n🚀 Training {model_name} with Grid Search...\n")

                grid_search = GridSearchCV(model, param_grid, scoring=metric, cv=StratifiedKFold(n_splits=3), n_jobs=-1)
                grid_search.fit(xTrain, yTrain)  # Find best parameters
                best_model = grid_search.best_estimator_
                models_dict[model_name] = best_model


            train_scores, val_scores = [], []
            train_accuracies, val_accuracies = [], []
            train_f1s, val_f1s = [], []
            train_briers, val_briers = [], []

            for train_index, test_index in cv.split(xTrain):
                X_train, X_test = xTrain[train_index], xTrain[test_index]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                # xVal = scaler.transform(xVal)
                y_train, y_test = yTrain[train_index], yTrain[test_index]


                if model_name == 'XGBoost':
                    best_model.fit(X_train,y_train)
                    calibrated_model = CalibratedClassifierCV(best_model,
                                                              method='sigmoid',cv=5)  # or 'sigmoid' for Platt Scaling
                    calibrated_model.fit(X_train, y_train)
                    y_train_pred = calibrated_model.predict(X_train)
                    y_val_pred = calibrated_model.predict(X_test)
                elif model_name == 'Stacked':
                    # Define base models
                    base_models = [
                        ('xgb',models_dict['XGBoost']),
                        ('logreg', models_dict['LogisticRegression']),
                        # ('SVC', SVC(random_state=13,probability=True)),
                        ('KNN', KNeighborsClassifier()),
                        ('Gaussian NB', GaussianNB())
                    ]

                    # Define meta-model (usually Logistic Regression)
                    stacking_model = StackingClassifier(
                        estimators=base_models,
                        final_estimator=Perceptron(),
                        stack_method='predict_proba'  # Use probability predictions
                    )

                    # Train Stacking Model
                    stacking_model.fit(X_train, y_train)
                    calibrated_model = CalibratedClassifierCV(stacking_model,
                                                              method='sigmoid', cv=5)  # or 'sigmoid' for Platt Scaling
                    calibrated_model.fit(X_train, y_train)
                    y_train_pred = calibrated_model.predict(X_train)
                    y_val_pred = calibrated_model.predict(X_test)
                elif model_name == 'Ensemble':
                    # Train and calibrate models
                    log_reg = models_dict['LogisticRegression'].fit(X_train, y_train)
                    xgb = models_dict['XGBoost'].fit(X_train, y_train)

                    # Get probability predictions
                    log_proba = log_reg.predict_proba(X_test)[:, 1]
                    xgb_proba = xgb.predict_proba(X_test)[:, 1]
                    log_proba_train = xgb.predict_proba(X_train)[:, 1]
                    xgb_proba_train = xgb.predict_proba(X_train)[:, 1]

                    # Define weights (adjust based on validation performance)
                    w_log = 0.6  # More weight to Logistic Regression (best Brier Score)
                    w_xgb = 0.4  # Less weight to XGBoost

                    # Weighted probability prediction
                    ensemble_proba = (w_log * log_proba) + (w_xgb * xgb_proba)
                    ensemble_proba_train = (w_log * log_proba_train) + (w_xgb * xgb_proba_train)

                    # Convert to final binary predictions
                    y_val_pred = (ensemble_proba > 0.5).astype(int)
                    y_train_pred = (ensemble_proba_train > 0.5).astype(int)
                else:
                    best_model.fit(X_train,y_train)
                    calibrated_model = CalibratedClassifierCV(best_model,
                                                              method='sigmoid', cv=5)  # or 'sigmoid' for Platt Scaling
                    calibrated_model.fit(X_train, y_train)
                    y_train_pred = calibrated_model.predict(X_train)
                    y_val_pred = calibrated_model.predict(X_test)

                # Compute the selected metric
                metric_func = {
                    "precision": precision_score,
                    "recall": recall_score,
                    "f1": f1_score,
                    "accuracy": accuracy_score,
                    "neg_brier_score": brier_score_loss
                }.get(metric,"")

                train_precision = precision_score(y_train, y_train_pred, zero_division=0)
                val_precision = precision_score(y_test, y_val_pred, zero_division=0)

                train_scores.append(train_precision)
                val_scores.append(val_precision)

                # Compute accuracy & F1 score
                train_acc = accuracy_score(y_train, y_train_pred)
                val_acc = accuracy_score(y_test, y_val_pred)
                train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
                val_f1 = f1_score(y_test, y_val_pred, zero_division=0)
                train_brier = brier_score_loss(y_train, y_train_pred)
                val_brier = brier_score_loss(y_test, y_val_pred)

                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                train_f1s.append(train_f1)
                val_f1s.append(val_f1)
                train_briers.append(train_brier)
                val_briers.append(val_brier)

                # Plot confusion matrix
                # cm = confusion_matrix(y_test, y_test_pred)
                # plt.figure(figsize=(5, 4))
                # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                # plt.xlabel("Predicted")
                # plt.ylabel("Actual")
                # plt.title(f"{model_name} Confusion Matrix")
                # plt.show()

                # Compute average scores
                avg_train_score = np.mean(train_scores)
                avg_val_score = np.mean(val_scores)
                avg_train_acc = np.mean(train_accuracies)
                avg_val_acc = np.mean(val_accuracies)
                avg_train_f1 = np.mean(train_f1s)
                avg_val_f1 = np.mean(val_f1s)
                avg_train_brier = np.mean(train_briers)
                avg_val_brier = np.mean(val_briers)

                # Save results
                results[model_name] = {
                    "train_scores": train_scores,
                    "val_scores": val_scores,
                    "avg_train_score": avg_train_score,
                    "avg_val_score": avg_val_score,
                    "train_accuracy": train_accuracies,
                    "val_accuracy": val_accuracies,
                    "avg_train_accuracy": avg_train_acc,
                    "avg_val_accuracy": avg_val_acc,
                    "train_f1": train_f1s,
                    "val_f1": val_f1s,
                    "avg_train_f1": avg_train_f1,
                    "avg_val_f1": avg_val_f1,
                    "avg_train_brier": avg_train_brier,
                    "avg_val_brier": avg_val_brier,
                    "best_params": grid_search.best_params_
                }

            # Print results
            # print(f"{model_name} - Train {metric.capitalize()}: {train_scores}")
            # print(f"{model_name} - Validation {metric.capitalize()}: {val_scores}")
            print(f"{model_name} - Average Train {metric.capitalize()}: {avg_train_score:.4f}")
            print(f"{model_name} - Average Validation {metric.capitalize()}: {avg_val_score:.4f}")
            # print(f"{model_name} - Train Accuracy: {train_accuracies}")
            # print(f"{model_name} - Validation Accuracy: {val_accuracies}")
            print(f"{model_name} - Average Train Accuracy: {avg_train_acc:.4f}")
            print(f"{model_name} - Average Validation Accuracy: {avg_val_acc:.4f}")
            # print(f"{model_name} - Train F1 Score: {train_f1s}")
            # print(f"{model_name} - Validation F1 Score: {val_f1s}")
            print(f"{model_name} - Average Train F1 Score: {avg_train_f1:.4f}")
            print(f"{model_name} - Average Validation F1 Score: {avg_val_f1:.4f}")
            print(f"{model_name} - Average Train Brier Score: {avg_train_brier:.4f}")
            print(f"{model_name} - Average Validation Brier Score: {avg_val_brier:.4f}")
            print(f"{model_name} - Best Hyperparameters: {grid_search.best_params_}")

            # Overfitting Detection
            if avg_train_brier < avg_val_brier * .75:
                print("🚨 Possible Overfitting: Training score is much higher than Validation score")
            elif avg_train_brier > avg_val_brier * 0.95:
                print("✅ Model seems well-generalized.")
            else:
                print("⚠️ Mild Overfitting: Small gap, but may be acceptable.")

            # Save the trained model
            model_filename = f"2025_saved_data/{self.gender}_{model_name}_classifier.mdl"
            with open(model_filename, 'wb') as outp:
                pickle.dump(best_model, outp, pickle.HIGHEST_PROTOCOL)

        return results

    def train_full_model(self):
        X,y = self.xTrain, self.yTrain

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        features = ['numWins', 'avgPoss', 'totalPointsPerPoss', 'totalOppPointsPerPoss', 'totalfgmPerPoss',
                    'totalOppFgmPerPoss', 'totalfgaPerPoss', 'totalOppFgaPerPoss', 'totalfgm3PerPoss',
                    'totalOppFgm3PerPoss', 'totalfga3PerPoss', 'totalOppFga3PerPoss', 'totalftmPerPoss',
                    'totalOppFtmPerPoss', 'totalftaPerPoss', 'totalOppFtaPerPoss', 'totalorPerPoss',
                    'totalOppOrPerPoss',
                    'totaldrPerPoss', 'totalOppDrPerPoss', 'totalastPerPoss', 'totalOppAstPerPoss', 'totaltoPerPoss',
                    'totalOppToPerPoss', 'totalstlPerPoss', 'totalOppStlPerPoss', 'totalblkPerPoss',
                    'totalOppBlkPerPoss',
                    'totalpfPerPoss', 'totalOppPfPerPoss', 'totalDrEff', 'totalOrEff', 'totalBlkEff', 'totalOppBlkEff',
                    'stlPerDr', 'oppStlPerDr', 'Power6Conference', 'tournamentSeed', 'ord_rank',
                    'TourneyAppearances', 'sos', 'srs', 'Quality_G_Count', 'Quality_F_Count', 'home']
        new_X, dropped_features, importance_df_nb, selected_mask = self.recursive_feature_elimination(
            X=X_scaled, y=y,
            feature_names=features,
            estimator=XGBClassifier())
        self.main_mask = selected_mask
        print(f"\n🚀 Training XGBoost with Grid Search...\n")

        grid_search = GridSearchCV(XGBClassifier(objective='binary:logistic'), {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1],
                 'reg_lambda': [0, .5, 1], 'reg_alpha': [0, 0.5, 1]}, scoring='accuracy', cv=StratifiedKFold(n_splits=3), n_jobs=-1)
        grid_search.fit(new_X, y)  # Find best parameters
        best_model = grid_search.best_estimator_
        best_model.fit(new_X, y)
        calibrated_model = CalibratedClassifierCV(best_model,
                                                  method='sigmoid', cv=5)  # or 'sigmoid' for Platt Scaling
        calibrated_model.fit(new_X, y)
        self.model = best_model
        self.calibrated_model = calibrated_model
        self.scaler = scaler
        model_filename = f"2025_saved_data/{self.gender}_main_classifier.mdl"
        with open(model_filename, 'wb') as outp:
            pickle.dump(calibrated_model, outp, pickle.HIGHEST_PROTOCOL)
        scaler_filename = f"2025_saved_data/{self.gender}_main_scaler.pkl"
        with open(scaler_filename, 'wb') as outp:
            pickle.dump(scaler, outp, pickle.HIGHEST_PROTOCOL)
    def get_upset_X_y(self, years: range):
        from tqdm import tqdm
        if os.path.exists(self.season_dict_file):
            with open(self.season_dict_file, "rb") as f:
                season_dicts = pickle.load(f)
            print("Loaded season_dicts from file.")
        else:
            season_dicts = {}
        totalNumGames = 0
        merged_tourney = self.tourney_compact_pd.merge(self.tourney_seeds_pd, left_on=['Season', 'WTeamID'],
                                       right_on=['Season', 'TeamID'], how='left')
        merged_tourney.rename(columns={'Seed': 'WTeamSeed'}, inplace=True)
        merged_tourney.drop(columns=['TeamID'], inplace=True)  # Drop redundant column
        # Merge to get LTeamSeed
        merged_tourney = merged_tourney.merge(self.tourney_seeds_pd, left_on=['Season', 'LTeamID'],
                                              right_on=['Season', 'TeamID'], how='left')
        merged_tourney.rename(columns={'Seed': 'LTeamSeed'}, inplace=True)
        merged_tourney.drop(columns=['TeamID'], inplace=True)  # Drop redundant column
        try:
            merged_tourney['WTeamSeed'] = merged_tourney['WTeamSeed'].apply(extract_number)
            merged_tourney['LTeamSeed'] = merged_tourney['LTeamSeed'].apply(extract_number)
        except:
            pass
        merged_tourney['SeedDiff'] = abs(merged_tourney['LTeamSeed'] - merged_tourney['WTeamSeed'])
        merged_tourney.loc[merged_tourney['WTeamSeed'] > merged_tourney['LTeamSeed'], 'upset'] = 1
        merged_tourney['upset'].fillna(0, inplace=True)  # Fill remaining values with 0
        merged_tourney['upset'] = merged_tourney['upset'].astype(int)

        merged_season = self.reg_season_compact_pd.merge(self.tourney_seeds_pd, left_on=['Season', 'WTeamID'],
                                                       right_on=['Season', 'TeamID'], how='left')
        merged_season.rename(columns={'Seed': 'WTeamSeed'}, inplace=True)
        merged_season.drop(columns=['TeamID'], inplace=True)  # Drop redundant column
        # Merge to get LTeamSeed
        merged_season = merged_season.merge(self.tourney_seeds_pd, left_on=['Season', 'LTeamID'],
                                              right_on=['Season', 'TeamID'], how='left')
        merged_season.rename(columns={'Seed': 'LTeamSeed'}, inplace=True)
        merged_season.drop(columns=['TeamID'], inplace=True)  # Drop redundant column
        merged_season['WTeamSeed'].fillna("W91",inplace=True)
        merged_season['LTeamSeed'].fillna("W91",inplace=True)
        try:
            merged_season['WTeamSeed'] = merged_season['WTeamSeed'].apply(extract_number)
            merged_season['LTeamSeed'] = merged_season['LTeamSeed'].apply(extract_number)
        except:
            pass
        merged_season['SeedDiff'] = abs(merged_season['LTeamSeed'] - merged_season['WTeamSeed'])
        merged_season.loc[merged_season['WTeamSeed'] > merged_season['LTeamSeed'], 'upset'] = 1
        merged_season['upset'].fillna(0, inplace=True)  # Fill remaining values with 0
        merged_season['upset'] = merged_season['upset'].astype(int)
        # loop through years
        for year in tqdm(years):
            if year==2020:
                continue
            season = self.filtered_regular[self.filtered_regular['Season'] == year]
            tourney = merged_tourney[merged_tourney['Season'] == year]

            season_trimmed = self.get_season_for_tourney_teams(merged_season, tourney)

            totalNumGames += len(season_trimmed.index)
            totalNumGames += len(tourney.index)
        try:
            numFeatures = len(
            self.getSeasonData(1181, 2012))  # Just choosing a random team and seeing the dimensionality of the vector
        except:
            numFeatures = len(self.getSeasonData(3163, 2014))
        x_Train = np.zeros((totalNumGames, numFeatures + 1))
        y_Train = np.zeros(totalNumGames)
        indexCounter = 0
        for year in tqdm(years):
            # team_vectors = self.load_or_create_season_dicts(year)
            # team_vectors = self.createSeasonDict(year)
            season_dicts[year] = season_dicts.get(year, self.createSeasonDict(year))
            team_vectors = season_dicts[year]
            # season = self.filtered_regular[self.filtered_regular['Season'] == year]
            tourney = merged_tourney[merged_tourney['Season'] == year]
            season_trimmed = self.get_season_for_tourney_teams(merged_season, tourney)
            # numGamesInSeason = len(season.index)
            numGamesInSeason = len(tourney.index)+len(season_trimmed.index)
            xTrainSeason = np.zeros((numGamesInSeason, numFeatures + 1))
            yTrainSeason = np.zeros((numGamesInSeason))
            counter = 0
            # for index, row in tourney.iterrows():
            #     w_team = row['WTeamID']
            #     w_vector = team_vectors[w_team]
            #     l_team = row['LTeamID']
            #     l_vector = team_vectors[l_team]
            #     diff = [a - b for a, b in zip(w_vector, l_vector)]
            #     home = self.getHomeStat(row['WLoc'])
            #     if (counter % 2 == 0):
            #         diff.append(home)
            #         xTrainSeason[counter] = diff
            #         yTrainSeason[counter] = 1
            #     else:
            #         diff.append(-home)
            #         xTrainSeason[counter] = [-p for p in diff]
            #         yTrainSeason[counter] = 0
            #     counter += 1
            for index, row in tqdm(tourney.iterrows()):
                w_team = row['WTeamID']
                w_vector = team_vectors[w_team]
                l_team = row['LTeamID']
                l_vector = team_vectors[l_team]
                if row['WTeamSeed'] <= row['LTeamSeed']:
                    diff = [a - b for a, b in zip(w_vector, l_vector)]
                else:
                    diff = [b-a for a,b, in zip(w_vector, l_vector)]
                home = 0  # All tournament games are neutral
                diff.append(home)
                xTrainSeason[counter] = diff
                yTrainSeason[counter] = row['upset']

                counter += 1
            for index, row in tqdm(season_trimmed.iterrows()):
                w_team = row['WTeamID']
                w_vector = team_vectors[w_team]
                l_team = row['LTeamID']
                l_vector = team_vectors[l_team]
                if row['WTeamSeed'] <= row['LTeamSeed']:
                    diff = [a - b for a, b in zip(w_vector, l_vector)]
                else:
                    diff = [b - a for a, b, in zip(w_vector, l_vector)]
                home = 0  # All tournament games are neutral
                diff.append(home)
                xTrainSeason[counter] = diff
                yTrainSeason[counter] = row['upset']

                counter += 1
            x_Train[indexCounter:numGamesInSeason + indexCounter] = xTrainSeason
            y_Train[indexCounter:numGamesInSeason + indexCounter] = yTrainSeason
            indexCounter += numGamesInSeason
        self.upset_X = x_Train
        self.upset_y = y_Train
        return x_Train, y_Train
    def createTrainingSet(self, years: range) -> Tuple[np.array, np.array]:
        """Creates a training set and a training target set

        This function iterates through a range of years and creates a training set of matchup vectors as well as a target
        vector indicating if the first named team in the matchup won or lost.

        Args:
            years (range): A range object of years to include in the training set
        Returns:
            Tuple[np.array,np.array]
        """
        from tqdm import tqdm
        if os.path.exists(self.season_dict_file):
            with open(self.season_dict_file, "rb") as f:
                season_dicts = pickle.load(f)
            print("Loaded season_dicts from file.")
        else:
            season_dicts = {}
        totalNumGames = 0
        # loop through years
        for year in tqdm(years):
            season = self.filtered_regular[self.filtered_regular['Season'] == year]
            tourney = self.tourney_compact_pd[self.tourney_compact_pd['Season'] == year]
            season_trimmed = self.get_season_for_tourney_teams(season, tourney)

            totalNumGames += len(season.index)
            totalNumGames += len(tourney.index)
        if self.gender == "M":
            numFeatures = len(
            self.getSeasonData(1181, 2012))
        else:
            numFeatures = len(
                self.getSeasonData(3290, 2012))
        x_Train = np.zeros((totalNumGames, numFeatures + 1))
        y_Train = np.zeros(totalNumGames)
        indexCounter = 0
        for year in tqdm(years):
            numGamesInSeason = 0
            season_dicts[year] = season_dicts.get(year,self.createSeasonDict(year))
            team_vectors = season_dicts[year]

            season = self.filtered_regular[self.filtered_regular['Season'] == year]
            tourney = self.tourney_compact_pd[self.tourney_compact_pd['Season'] == year]
            season_trimmed = self.get_season_for_tourney_teams(season, tourney)
            numGamesInSeason += len(season.index)
            numGamesInSeason += len(tourney.index)
            xTrainSeason = np.zeros((numGamesInSeason, numFeatures + 1))
            yTrainSeason = np.zeros((numGamesInSeason))
            counter = 0
            for index, row in tqdm(season.iterrows()):
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
            self.tourney_start_index = counter
            for index, row in tqdm(tourney.iterrows()):
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
        with open(self.season_dict_file, "wb") as f:
            pickle.dump(season_dicts, f)
        print("Saved season_dicts to file.")

        return x_Train, y_Train
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
            np.save(f'2025_saved_data/{self.gender}_xTrain', self.xTrain)
            np.save(f'2025_saved_data/{self.gender}_yTrain', self.yTrain)

        else:
            self.xTrain = np.load(f'2025_saved_data/{self.gender}_xTrain.npy')
            self.yTrain = np.load(f'2025_saved_data/{self.gender}_yTrain.npy')

        return self.xTrain, self.yTrain

    def train_upset_model(self):
        features = ['numWins', 'avgPoss', 'totalPointsPerPoss', 'totalOppPointsPerPoss', 'totalfgmPerPoss',
                    'totalOppFgmPerPoss', 'totalfgaPerPoss', 'totalOppFgaPerPoss', 'totalfgm3PerPoss',
                    'totalOppFgm3PerPoss', 'totalfga3PerPoss', 'totalOppFga3PerPoss', 'totalftmPerPoss',
                    'totalOppFtmPerPoss', 'totalftaPerPoss', 'totalOppFtaPerPoss', 'totalorPerPoss',
                    'totalOppOrPerPoss',
                    'totaldrPerPoss', 'totalOppDrPerPoss', 'totalastPerPoss', 'totalOppAstPerPoss', 'totaltoPerPoss',
                    'totalOppToPerPoss', 'totalstlPerPoss', 'totalOppStlPerPoss', 'totalblkPerPoss',
                    'totalOppBlkPerPoss',
                    'totalpfPerPoss', 'totalOppPfPerPoss', 'totalDrEff', 'totalOrEff', 'totalBlkEff', 'totalOppBlkEff',
                    'stlPerDr', 'oppStlPerDr', 'Power6Conference', 'tournamentSeed', 'ord_rank',
                    'TourneyAppearances', 'sos', 'srs', 'Quality_G_Count', 'Quality_F_Count', 'home']
        # upsetX = np.load('upsetX.npy')
        # upsety = np.load('upsety.npy')
        # self.upsetX = upsetX
        # self.upsety = upsety
        new_upset_X, dropped_features, importance_df_nb, selected_mask = self.recursive_feature_elimination(X=self.upset_X, y=self.upset_y,
                                                                              feature_names=features,
                                                                                            estimator=LinearSVC())
        self.upset_mask = selected_mask
        print(dropped_features)
        model = LogisticRegression(solver='liblinear', C=1)
        model.fit(new_upset_X, self.upset_y)
        self.upset_model = model
        with open(f'2025_saved_data/{self.gender}_upset_model.mdl', 'wb') as outp:
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
def extract_number(s):
    try:
        match = re.search(r'\d+', s)  # Find the first sequence of digits
    except:
        pass
    return int(match.group()) if match else None
# men = Data()
# men.read_data('men')
print("running")
import pickle
# with open('men_data_obj.pkl', 'wb') as out:
#     pickle.dump(men)
# with open('men_data_obj.pkl', 'rb') as inp:
#     men = pickle.load(inp)
# with open('XGBoost_WAS_model.mdl', 'rb') as inp:
#     model = pickle.load(inp)
# upset_X,upset_y = men.get_upset_X_y(range(2003,2024))
# np.save('upsetX',upset_X)
# np.save('upsety',upset_y)
# upsetX = np.load('upsetX.npy')
# upsety = np.load('upsety.npy')
# men.train_upset_model()
# with open('men_data_obj.pkl', 'wb') as out:
#     pickle.dump(men,out)
def remove_outliers_iqr(X,y):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
    return X[mask], y[mask]  # Keep only inliers

def save_data_obj(data_obj, file_name):
    with open(f'2025_saved_data/{file_name}.pkl', 'wb') as out:
        pickle.dump(data_obj, out)

def load_data_obj(file_name):
    with open(f'2025_saved_data/{file_name}.pkl', 'rb') as inp:
        obj = pickle.load(inp)
        return obj

def predict(id1,id2,year, data_obj):
    vector = data_obj.compareTwoTeams(id1, id2, year)
    vector.append(0)
    vector = np.array(vector)
    scaled_vector = data_obj.scaler.transform(vector.reshape(1,-1))

    selected_vector = scaled_vector.flatten()[data_obj.main_mask]
    prediction = data_obj.calibrated_model.predict_proba(selected_vector.reshape(1,-1))
    # print(prediction[0][1])
    return prediction

def predict_upset(id1,id2,year, data_obj):
    vector = data_obj.compareTwoTeams(id1, id2, year)
    vector.append(0)
    vector = np.array(vector)
    # scaled_vector = data_obj.scaler.transform(vector.reshape(1, -1))

    selected_vector = vector.flatten()[data_obj.upset_mask]
    prediction = data_obj.upset_model.predict_proba(selected_vector.reshape(1, -1))
    # print(prediction[0][1])
    return prediction

# X_cleaned, upsety = remove_outliers_iqr(upsetX,upsety)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_cleaned)

# new_upset_X, dropped_features, importance_df_rf = men.recursive_feature_elimination(X=X_scaled,y=upsety,feature_names=features,estimator=RandomForestClassifier())
# print(dropped_features)
# men.train_classifiers(new_upset_X,upsety)
# new_upset_X, dropped_features, importance_df_nb = men.recursive_feature_elimination(X=upsetX,y=upsety,feature_names=features,estimator=LinearSVC())
# print(dropped_features)

# men.train_classifiers(new_upset_X,upsety)
# # men.train_upset(estimator = LogisticRegression('C': 1))
# new_upset_X, dropped_features,importance_df_svm = men.recursive_feature_elimination(X=X_scaled,y=upsety,feature_names=features,estimator=GaussianNB())
# print(dropped_features)
# men.train_classifiers(new_upset_X, upsety)




# X,y = men.createTrainingSet(range(2003, 2024))
# X=np.load('X.npy')
# y=np.load('y.npy')
# season_dicts = {}
# with open('season_dicts.pkl', 'rb') as inp:
#     season_dicts = pickle.load(inp)
# # last_year = max(season_dicts.keys())
# # for year in range(last_year+1,2025):
# #     season_dicts[year] = men.createSeasonDict(year)
# # season_dicts = {k: v for k, v in season_dicts.items() if not (1993 <= k <= 2002)}
# # with open('season_dicts.pkl', 'wb') as out:
# #     pickle.dump(season_dicts,out)
# # for k,v in season_dicts.items():
# #     for team, vector in v.items():
# #         WAS = model.predict(np.array(vector).reshape(1,-1))[0]
#
#
# X = np.load('mwasX.npy')
# y = np.load('mwasy.npy')
#
# men.WAS_X = X
# men.WAS_y = y
# model = men.train_WAS_model()
# men.get_x_and_y(False,1993,2025)
#
# men = Data()
# men.read_data('men')
# #
# # with open('2025_saved_data/women_data_obj.pkl', 'rb') as inp:
# #     women = pickle.load(inp)
# upset_X,upset_y = men.get_upset_X_y(range(2003,2024))
# np.save('M_upsetX',upset_X)
# np.save('M_upsety',upset_y)
# men.train_upset_model()
# men.get_x_and_y(False, 2003,2024)
# with open('2025_saved_data/men_data_obj.pkl', 'wb') as out:
#     pickle.dump(men, out)
# #
# women = Data()
# women.read_data('women')
#
# # with open('2025_saved_data/women_data_obj.pkl', 'rb') as inp:
# #     women = pickle.load(inp)
# upset_X,upset_y = women.get_upset_X_y(range(2010,2024))
# np.save('W_upsetX',upset_X)
# np.save('W_upsety',upset_y)
# women.train_upset_model()
# women.get_x_and_y(False, 2010,2024)
# with open('2025_saved_data/women_data_obj.pkl', 'wb') as out:
#     pickle.dump(women, out)
# pass
# with open('2025_saved_data/men_data_obj.pkl', 'rb') as inp:
#     men = pickle.load(inp)
retrain = False
if retrain:
    men = Data()
    men.read_data('men')
    save_data_obj(men,'men_data_obj_read')
    # men = load_data_obj('men_data_obj_get_upset_x_y_whole')
    men.get_x_and_y(False, 2003,2026)
    save_data_obj(men, 'men_data_obj_get_x_y_whole')
    men.get_upset_X_y(range(2003, 2026))
    save_data_obj(men, 'men_data_obj_get_upset_x_y_whole')
    men.train_upset_model()
    # men.get_x_and_y(False, 2003, 2024)
    # save_data_obj(men, 'men_data_obj_get_x_y_top_teams')
    # men.get_x_and_y(False, 2003, 2026)
    # save_data_obj(men, 'men_data_obj_get_x_y_whole')
else:
    men = load_data_obj('men_data_obj_final')
    # men.train_upset_model()
    # men.train_full_model()
    # save_data_obj(men,'men_data_obj_final')
    # men.get_x_and_y(False, 2003, 2024)
    # save_data_obj(men, 'men_data_obj_get_x_y_top_teams')
# men.train_classifiers(men.xTrain, men.yTrain, metric='accuracy')

if retrain:
    women = Data()
    women.read_data('women')
    save_data_obj(women,'women_data_obj_read')
    # women = load_data_obj('women_data_obj_read')
    women.get_x_and_y(False, 2010,2026)
    save_data_obj(women, 'women_data_obj_get_x_y_whole')
    women.get_upset_X_y(range(2010, 2026))
    save_data_obj(women, 'women_data_obj_get_upset_x_y_whole')
    women.train_upset_model()
    save_data_obj(women, 'women_data_obj_final')
    # women.get_x_and_y(False, 2010,2024)
    # save_data_obj(women, 'women_data_obj_get_x_y_top_teams')
    # women.get_x_and_y(False, 2010,2024)
    # save_data_obj(women, 'women_data_obj_get_x_y_tourney')
else:
    # women = Data()
    # women.read_data('women')
    women = load_data_obj('women_data_obj_final')
    # women.train_upset_model()
    # save_data_obj(women, 'women_data_obj_final')    # women.get_x_and_y(False, 2010,2026)
    # save_data_obj(women, 'women_data_obj_get_x_y_whole')
    # women.get_upset_X_y(range(2010, 2026))
    # save_data_obj(women, 'women_data_obj_get_upset_x_y_whole')
    # women.train_upset_model()
    # women.train_full_model()
    # save_data_obj(women, 'women_data_obj_final')
    # save_data_obj(women, 'women_data_obj_get_x_y_whole')
# women.train_classifiers(women.xTrain, women.yTrain, metric='accuracy')
# X_train, X_test, y_train, y_test = train_test_split(men.xTrain, men.yTrain, test_size=0.2)
# svc_model = RandomForestClassifier()
# svc_model.fit(X_train, y_train)
# prob=svc_model.predict_proba(X_test)[:, 1]
# from sklearn.calibration import calibration_curve
# x_p, y_p=calibration_curve(y_test, prob, n_bins=10)
# plt.plot([0, 1], [0, 1])
# plt.plot(x_p, y_p)
# plt.show()
# from sklearn.metrics import brier_score_loss, roc_auc_score
#
# y_pred = svc_model.predict(X_test)
#
# print(brier_score_loss(y_test, y_pred, pos_label=2))
#
# print(roc_auc_score(y_test, y_pred))
#
# from sklearn.calibration import CalibratedClassifierCV
# calib_model = CalibratedClassifierCV(svc_model, method='sigmoid', cv=5)
# calib_model.fit(X_train, y_train)
# prob = calib_model.predict_proba(X_test)[:, 1]
#
# #plot the calibration curve (see above to know what is it)
# x_p, y_p = calibration_curve(y_test, prob, n_bins=10)
# plt.plot([0, 1], [0, 1])
# plt.plot(x_p, y_p)
# plt.show()
#
# y_pred = calib_model.predict(X_test)
# print(brier_score_loss(y_test, y_pred, pos_label=2))
# print(roc_auc_score(y_test, y_pred))

M_tourney_teams = pd.read_csv('files/MNCAATourneySeeds.csv')
W_tourney_teams = pd.read_csv('files/WNCAATourneySeeds.csv')
M_2025_tourney_teams = list(M_tourney_teams[M_tourney_teams['Season']==2025]['TeamID'])
W_2025_tourney_teams = list(W_tourney_teams[W_tourney_teams['Season']==2025]['TeamID'])
# Load the CSV
MTeamsdf = pd.read_csv('files/MTeamSpellings.csv')  # Replace with actual filename
WTeamsdf = pd.read_csv('files/WTeamSpellings.csv')  # Replace with actual filename

def get_team_id(team_name, gender):
    if gender == "M":
        row = MTeamsdf[MTeamsdf['TeamNameSpelling'].str.lower() == team_name.lower()]
        return row['TeamID'].iloc[-1] if not row.empty else None
    elif gender == "W":
        row = WTeamsdf[WTeamsdf['TeamNameSpelling'].str.lower() == team_name.lower()]
        return row['TeamID'].iloc[-1] if not row.empty else None
def compute_prediction(row, mteams,wteams):
    team_split = row[0].split("_")
    team1, team2 = int(team_split[1]), int(team_split[2])
    if (team1 in mteams and team2 in mteams):
        return (predict(team1, team2, 2025, men)[0][1] + predict(team2, team1, 2025, men)[0][0]) / 2
    if (team1 in wteams and team2 in wteams):
        return (predict(team1, team2, 2025, women)[0][1] + predict(team2, team1, 2025, women)[0][0]) / 2
    return row[1]

def create_submission(mteams,wteams):
    df = pd.read_csv('files/SampleSubmissionStage2.csv')
    df[df.columns[1]] = df.apply(lambda row: compute_prediction(row, mteams, wteams), axis=1)
    df.to_csv('2025_saved_data/submission.csv')
    # for index, row in tqdm(df.iterrows()):
    #     team_split = row[0].split("_")
    #     team1, team2 = int(team_split[1]), int(team_split[2])
    #     if (team1 in mteams and team2 in mteams):
    #         prediction = (predict(team1, team2, 2025, men)[0][1] + predict(team2, team1, 2025, men)[0][0]) / 2
    #         row[1] = prediction
# create_submission(M_2025_tourney_teams,[])

def predict_by_name(team1,team2,data_obj):
    team1_id = get_team_id(team1,data_obj.gender)
    team2_id = get_team_id(team2,data_obj.gender)
    prediction = (predict(team1_id, team2_id, 2025, data_obj)[0][1] + predict(team2_id, team1_id, 2025, data_obj)[0][0]) / 2
    upset_prediction = (predict_upset(team1_id, team2_id, 2025, data_obj)[0][1] + predict_upset(team2_id, team1_id, 2025, data_obj)[0][0]) / 2
    team1_gs, team1_fs = data_obj.get_key_players(team1_id,2025)
    team2_gs, team2_fs = data_obj.get_key_players(team2_id,2025)
    print(f'{team1} has a {prediction*100}% chance of beating {team2}')
    print(f'{team1} vs {team2} has a {upset_prediction*100}% chance of being an upset')
    print(f"Key players: {team1} - {team1_gs} guards and {team1_fs} forwards")
    print(f"Key players: {team2} - {team2_gs} guards and {team2_fs} forwards")
men_seed_result_map = {}
for index, row in men.tourney_compact_pd.iterrows():
    Season = row['Season']
    WTeamID = row['WTeamID']
    WTeamSeed = \
    men.tourney_seeds_pd[(men.tourney_seeds_pd['Season'] == Season) & (men.tourney_seeds_pd['TeamID'] == WTeamID)][
        'Seed'].iloc[0]
    LTeamID = row['LTeamID']
    LTeamSeed = \
    men.tourney_seeds_pd[(men.tourney_seeds_pd['Season'] == Season) & (men.tourney_seeds_pd['TeamID'] == LTeamID)][
        'Seed'].iloc[0]
    WTeamSeed = extract_number(WTeamSeed)
    LTeamSeed = extract_number(LTeamSeed)
    seed_diff = abs(WTeamSeed-LTeamSeed)
    if WTeamSeed < LTeamSeed:
        men_seed_result_map[seed_diff] = men_seed_result_map.get(seed_diff,[0,0])
        men_seed_result_map[seed_diff][0] += 1
        men_seed_result_map[seed_diff][1] += 1
    else:
        men_seed_result_map[seed_diff] = men_seed_result_map.get(seed_diff,[0,0])
        men_seed_result_map[seed_diff][0] += 1
for key, value in men_seed_result_map.items():
    men_seed_result_map[key] = value[1]/value[0]

women_seed_result_map = {}
for index, row in women.tourney_compact_pd.iterrows():
    Season = row['Season']
    WTeamID = row['WTeamID']
    WTeamSeed = \
    women.tourney_seeds_pd[(women.tourney_seeds_pd['Season'] == Season) & (women.tourney_seeds_pd['TeamID'] == WTeamID)][
        'Seed'].iloc[0]
    LTeamID = row['LTeamID']
    LTeamSeed = \
    women.tourney_seeds_pd[(women.tourney_seeds_pd['Season'] == Season) & (women.tourney_seeds_pd['TeamID'] == LTeamID)][
        'Seed'].iloc[0]
    WTeamSeed = extract_number(WTeamSeed)
    LTeamSeed = extract_number(LTeamSeed)
    seed_diff = abs(WTeamSeed-LTeamSeed)
    if WTeamSeed < LTeamSeed:
        women_seed_result_map[seed_diff] = women_seed_result_map.get(seed_diff,[0,0])
        women_seed_result_map[seed_diff][0] += 1
        women_seed_result_map[seed_diff][1] += 1
    else:
        women_seed_result_map[seed_diff] = women_seed_result_map.get(seed_diff,[0,0])
        women_seed_result_map[seed_diff][0] += 1
for key, value in women_seed_result_map.items():
    women_seed_result_map[key] = value[1]/value[0]
men_seed_result_map[14] = (men_seed_result_map[13]+men_seed_result_map[15])/2
women_seed_result_map[14] = (women_seed_result_map[13]+women_seed_result_map[15])/2

def get_pred(teamid1,teamid2):
    if teamid1 < 2000:
        team1Seed = men.tourney_seeds_pd[(men.tourney_seeds_pd['Season'] == 2025) & (men.tourney_seeds_pd['TeamID'] == teamid1)][
        'Seed'].iloc[0]
        team2Seed = men.tourney_seeds_pd[(men.tourney_seeds_pd['Season'] == 2025) & (men.tourney_seeds_pd['TeamID'] == teamid2)][
        'Seed'].iloc[0]
        team1Seed = extract_number(team1Seed)
        team2Seed = extract_number(team2Seed)
        seed_diff = abs(team1Seed - team2Seed)
        if team1Seed < team2Seed:
            return men_seed_result_map[seed_diff]
        elif team2Seed < team1Seed:
            return 1 - men_seed_result_map[seed_diff]
        else:
            return .5
    else:
        team1Seed = women.tourney_seeds_pd[(women.tourney_seeds_pd['Season'] == Season) & (women.tourney_seeds_pd['TeamID'] == teamid1)][
        'Seed'].iloc[0]
        team2Seed = women.tourney_seeds_pd[(women.tourney_seeds_pd['Season'] == Season) & (women.tourney_seeds_pd['TeamID'] == teamid2)][
        'Seed'].iloc[0]
        team1Seed = extract_number(team1Seed)
        team2Seed = extract_number(team2Seed)
        seed_diff = abs(team1Seed - team2Seed)
        if team1Seed < team2Seed:
            return women_seed_result_map[seed_diff]
        elif team2Seed < team1Seed:
            return 1 - women_seed_result_map[seed_diff]
        else:
            return .5
while True:
    team1 = input("First team: ")
    team2 = input("Second name: ")
    team1_id = get_team_id(team1, men.gender)
    team2_id = get_team_id(team2, men.gender)
    predict_by_name(team1,team2,men)
    print(get_pred(team1_id,team2_id)*100)
# create_submission(M_2025_tourney_teams,W_2025_tourney_teams)

#
#
# input_file = 'files/SampleSubmissionStage2.csv'
# output_file = '2025_saved_data/submission2.csv'
#
# with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)
#     header = next(reader, None)  # Skip the first row (header)
#     if header:
#         writer.writerow(header)
#     for row in reader:
#         if not row:  # Skip empty lines
#             continue
#         try:
#             team1 = int(row[0].split("_")[1])
#             team2 = int(row[0].split("_")[2])
#             prediction = get_pred(team1, team2)
#             writer.writerow([row[0], prediction])
#         except (IndexError, ValueError) as e:
#             writer.writerow([row[0],.5])