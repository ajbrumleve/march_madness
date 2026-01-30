import random
from tqdm import tqdm
import pandas as pd

import march_madness


class Region:
    def __init__(self, region, gender, MMO, season_dict, matchup_preds,gender_teams):

        self.round_results = {}
        self.region = region
        self.gender = gender
        self.MMO = MMO
        self.season_dict = season_dict
        self.matchup_preds = matchup_preds
        self.path = {f"{region}01": f"R1{region}1",
                     f"{region}16": f"R1{region}1",
                     f"{region}08": f"R1{region}8",
                     f"{region}09": f"R1{region}8",
                     f"{region}05": f"R1{region}5",
                     f"{region}12": f"R1{region}5",
                     f"{region}04": f"R1{region}4",
                     f"{region}13": f"R1{region}4",
                     f"{region}06": f"R1{region}6",
                     f"{region}11": f"R1{region}6",
                     f"{region}03": f"R1{region}3",
                     f"{region}14": f"R1{region}3",
                     f"{region}07": f"R1{region}7",
                     f"{region}10": f"R1{region}7",
                     f"{region}02": f"R1{region}2",
                     f"{region}15": f"R1{region}2",
                     f"R1{region}1": f"R2{region}1",
                     f"R1{region}8": f"R2{region}1",
                     f"R1{region}5": f"R2{region}4",
                     f"R1{region}4": f"R2{region}4",
                     f"R1{region}6": f"R2{region}3",
                     f"R1{region}3": f"R2{region}3",
                     f"R1{region}7": f"R2{region}2",
                     f"R1{region}2": f"R2{region}2",
                     f"R2{region}1": f"R3{region}1",
                     f"R2{region}4": f"R3{region}1",
                     f"R2{region}3": f"R3{region}2",
                     f"R2{region}2": f"R3{region}2",
                     f"R3{region}1": f"R4{region}1",
                     f"R3{region}2": f"R4{region}1"}

        if self.region == "W" or self.region == "X":
            self.path[f"R4{region}1"] = "R5WX"
            self.path["R5WX"] = "R6CH"
        elif self.region == "Y" or self.region == "Z":
            self.path[f"R4{region}1"] = "R5YZ"
            self.path["R5YZ"] = "R6CH"
        self.teams = self.get_region_teams(gender_teams)

    def get_matchups(self, round_num):
        round_teams = [key for key, value in self.path.items() if value.startswith(f'R{round_num}')]
        matching_team_pairs = {}
        for team in round_teams:
            matching_team = [(key, value) for key, value in self.path.items() if
                             value == self.path[team] and key != team]
            if matching_team:
                matching_team_pairs[matching_team[0][1]] = (team, matching_team[0][0])
                round_teams.remove(team)
        return matching_team_pairs

    def get_region_teams(self,gender_teams):
        region_teams = gender_teams[gender_teams["Seed"].str[0] == self.region]
        region_teams_dict = region_teams.set_index('Seed')['TeamID'].to_dict()
        return region_teams_dict

    def process_matchups(self, round_num):
        matchups = self.get_matchups(round_num)
        for key, value in matchups.items():
            team_1, team_2 = value
            if round_num == 1:
                team1id = self.teams[team_1]
                team2id = self.teams[team_2]
            else:
                team_1 = self.teams[team_1]
                team_2 = self.teams[team_2]
                team1id = self.teams[team_1]
                team2id = self.teams[team_2]
            if (team1id,team2id) in self.matchup_preds[round_num]:
                team_1_prob = self.matchup_preds[round_num][(team1id,team2id)]
            else:
                vector_1 = self.season_dict[team1id]
                vector_2 = self.season_dict[team2id]
                team_1_prob = self.MMO.predictGame(vector_1, vector_2, 0)
                self.matchup_preds[round_num][(team1id,team2id)] = team_1_prob
            # team_1_prob = random.random()
            if random.random() <= team_1_prob:
                self.teams[key] = team_1
            else:
                self.teams[key] = team_2
        if round_num == 4:
            self.champion = self.teams[f"R4{self.region}1"]


class FinalFour:
    def __init__(self, gender, MMO, season_dict, matchup_preds,gender_teams):
        self.matchup_preds = matchup_preds
        self.gender = gender
        self.MMO = MMO
        self.season_dict = season_dict
        self.path = {"R4W1": "R5WX",
                     "R4X1": "R5WX",
                     "R4Y1": "R5YZ",
                     "R4Z1": "R5YZ",
                     "R5WX": "R6CH",
                     "R5YZ": "R6CH"
                     }
        self.teams = self.get_region_teams(gender_teams)

    def get_matchups(self, round_num):
        round_teams = [key for key, value in self.path.items() if value.startswith(f'R{round_num}')]
        matching_team_pairs = {}
        for team in round_teams:
            matching_team = [(key, value) for key, value in self.path.items() if
                             value == self.path[team] and key != team]
            if matching_team:
                matching_team_pairs[matching_team[0][1]] = (team, matching_team[0][0])
                round_teams.remove(team)
        return matching_team_pairs

    def get_region_teams(self, gender_teams):
        teams = pd.read_csv('2024_tourney_seeds.csv')
        gender_teams = teams[teams["Tournament"] == self.gender]

        region_teams_dict = gender_teams.set_index('Seed')['TeamID'].to_dict()
        return region_teams_dict
    def process_matchups(self, round_num):
        matchups = self.get_matchups(round_num)
        for key, value in matchups.items():
            team_1, team_2 = value
            team_1 = self.teams[team_1]
            team_2 = self.teams[team_2]
            team1id = self.teams[team_1]
            team2id = self.teams[team_2]
            if (team1id,team2id) in self.matchup_preds[round_num]:
                team_1_prob = self.matchup_preds[round_num][(team1id,team2id)]
            else:
                vector_1 = self.season_dict[team1id]
                vector_2 = self.season_dict[team2id]
                team_1_prob = self.MMO.predictGame(vector_1, vector_2, 0)
                self.matchup_preds[round_num][(team1id,team2id)] = team_1_prob

            # team_1_prob = random.random()
            if random.random() <= team_1_prob:
                self.teams[key] = team_1
            else:
                self.teams[key] = team_2


sample_submission = pd.read_csv("sample_submission.csv")


def get_play_in_games(season_dict,MMO, gender):
    if gender =="M":
        teams = pd.read_csv('files/MNCAATourneySeeds.csv')
    else:
        teams = pd.read_csv('files/WNCAATourneySeeds.csv')
    slot_dict = {}
    play_in_teams = teams[
        (teams['Season'] == 2024) & (teams['Seed'].str.contains("a") | teams['Seed'].str.contains("b"))]
    for i in range(4):
        team1_id = play_in_teams['TeamID'].iloc[i * 2]
        team2_id = play_in_teams['TeamID'].iloc[(i * 2) + 1]
        slot = play_in_teams['Seed'].iloc[i * 2][:-1]
        vector_1 = season_dict[team1_id]
        vector_2 = season_dict[team2_id]
        team_1_prob = MMO.predictGame(vector_1, vector_2, 0)

        if random.random() < team_1_prob:
            slot_dict[slot] = team1_id
        else:
            slot_dict[slot] = team2_id
        if gender == "M":
            slot_dict["X16"] = 1447
            slot_dict["Y10"] = 1161
            slot_dict["Y16"] = 1212
            slot_dict['Z10'] = 1160
        else:
            slot_dict["W16"] = 3342
            slot_dict["Z12"] = 3435
    return slot_dict
def run_simulation(gender, data_obj,season_dict,matchup_preds,gender_teams):
    submission = sample_submission.copy()
    slot_dict = get_play_in_games(season_dict,data_obj,gender)
    for index, row in gender_teams.iterrows():
        seed = row['Seed']
        if seed in slot_dict:
            gender_teams.at[index, 'TeamID'] = slot_dict[seed]
    submission_gender = submission[submission['Tournament'] == gender]
    matchup_dict = {}
    w = Region("W", gender, data_obj, season_dict, matchup_preds, gender_teams)
    x = Region("X", gender, data_obj,season_dict, matchup_preds,gender_teams)
    y = Region("Y", gender, data_obj,season_dict, matchup_preds,gender_teams)
    z = Region("Z", gender, data_obj,season_dict, matchup_preds,gender_teams)

    ff = FinalFour(gender, data_obj, season_dict, matchup_preds,gender_teams)
    for i in range(4):
        w.process_matchups(i + 1)
        x.process_matchups(i + 1)
        y.process_matchups(i + 1)
        z.process_matchups(i + 1)
    ff.teams["R4W1"] = w.champion
    ff.teams["R4X1"] = x.champion
    ff.teams["R4Y1"] = y.champion
    ff.teams["R4Z1"] = z.champion
    ff.process_matchups(5 )
    ff.process_matchups(6)
    matchup_dict.update(w.teams)
    matchup_dict.update(x.teams)
    matchup_dict.update(y.teams)
    matchup_dict.update(z.teams)
    matchup_dict.update(ff.teams)
    submission_gender['Team'] = submission_gender['Slot'].map(matchup_dict)
    return submission_gender, matchup_preds


def generate_portfolio(num_preds,m_data_obj,w_data_obj,season_dict):
    sub = pd.DataFrame()
    teams = pd.read_csv('2024_tourney_seeds.csv')
    m_teams = teams[teams["Tournament"] == "M"]
    w_teams = teams[teams["Tournament"] == "W"]
    matchup_preds = {1:{},2:{},3:{},4:{},5:{},6:{}}
    for i in tqdm(range(num_preds),colour='green'):
        mens_sub, matchup_preds = run_simulation("M",m_data_obj,season_dict,matchup_preds,m_teams)
        womens_sub, matchup_preds = run_simulation("W",w_data_obj,season_dict,matchup_preds,w_teams)
        mens_sub['Bracket'] = i + 1
        womens_sub['Bracket'] = i + 1
        sub = pd.concat([sub, mens_sub, womens_sub], ignore_index=True)
    sub = sub.reset_index()  # Reset the index to move it to a regular column
    sub.drop(columns=['RowId'], inplace=True, axis=0)  # Drop the existing 'RowID' column
    sub.rename(columns={'index': 'RowId'}, inplace=True)
    return sub


def generate_portfolio_new(num_preds, m_data_obj, w_data_obj, season_dict):
    teams = pd.read_csv('2024_tourney_seeds.csv')
    m_teams = teams[teams["Tournament"] == "M"]
    w_teams = teams[teams["Tournament"] == "W"]
    matchup_preds = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}

    # Preallocate DataFrame
    num_simulations_per_pred = 126
    total_rows = num_preds * num_simulations_per_pred
    sub = pd.DataFrame(index=range(total_rows), columns=['RowId', 'Tournament', 'Bracket', 'Slot', 'Team'])

    idx = 0
    for i in tqdm(range(num_preds), colour='green'):
        mens_sub, matchup_preds = run_simulation("M", m_data_obj, season_dict, matchup_preds, m_teams)
        womens_sub, matchup_preds = run_simulation("W", w_data_obj, season_dict, matchup_preds, w_teams)
        mens_sub['Bracket'] = i + 1
        womens_sub['Bracket'] = i + 1

        # Store DataFrames in a list for concatenation
        dfs_to_concat = [mens_sub, womens_sub]

        # Concatenate DataFrames
        for df in dfs_to_concat:
            num_rows = len(df)
            sub.iloc[idx:idx + num_rows] = df.values
            idx += num_rows

    # Drop rows with missing values
    sub.dropna(inplace=True)

    # Reset index
    sub.reset_index(drop=True, inplace=True)

    return sub
# import pickle
# # men_data_obj = march_madness.Data()
# # women_data_obj = march_madness.Data()
# with open('men_noseed_xgb_2024_march_madness_3.dill','rb') as file:
#     men_data_obj = pickle.load(file)
# with open('women_noseed_xgb_2024_march_madness_3.dill','rb') as file:
#     women_data_obj = pickle.load(file)
# with open('season_dict.dct','rb') as file:
#     season_dict = pickle.load(file)
# sub = generate_portfolio(1000,men_data_obj, women_data_obj,season_dict)
