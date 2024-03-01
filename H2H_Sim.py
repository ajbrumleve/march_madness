import random

import pandas as pd


class Region:
    def __init__(self, region, gender):

        self.round_results = {}
        self.region = region
        self.gender = gender
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
        self.teams = self.get_region_teams()

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

    def get_region_teams(self):
        teams = pd.read_csv('2024_tourney_seeds.csv')
        gender_teams = teams[teams["Tournament"] == self.gender]
        region_teams = gender_teams[gender_teams["Seed"].str[0] == self.region]
        region_teams_dict = region_teams.set_index('Seed')['TeamID'].to_dict()
        return region_teams_dict

    def process_matchups(self, round_num):
        matchups = self.get_matchups(round_num)
        for key, value in matchups.items():
            team_1, team_2 = value
            if round_num == 1:
                team1id = team_1
                team2id = team_2
            else:
                team1id = self.teams[team_1]
                team2id = self.teams[team_2]
            # vector_1 = march_madness_data_object.getSeasonData(team1id,2024)
            # vector_2 = march_madness_data_object.getSeasonData(team2id,2024)
            # team_1_prob = march_madness_data_object.predictGame(vector_1,vector_2,0)
            team_1_prob = random.random()
            if random.random() <= team_1_prob:
                self.teams[key] = team1id
            else:
                self.teams[key] = team2id
        if round_num == 4:
            self.champion = self.teams[f"R4{self.region}1"]

class FinalFour:
    def __init__(self, gender):
        self.gender = gender
        self.path = {"R4W1": "R5WX",
                     "R4X1": "R5WX",
                     "R4Y1": "R5YZ",
                     "R4Z1": "R5YZ",
                     "R5WX": "R6CH",
                     "R5YZ": "R6CH"
                     }
        self.teams = {}

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

    def process_matchups(self, round_num):
        matchups = self.get_matchups(round_num)
        for key, value in matchups.items():
            team_1, team_2 = value
            team1id = self.teams[team_1]
            team2id = self.teams[team_2]
            # vector_1 = march_madness_data_object.getSeasonData(team1id,2024)
            # vector_2 = march_madness_data_object.getSeasonData(team2id,2024)
            # team_1_prob = march_madness_data_object.predictGame(vector_1,vector_2,0)
            team_1_prob = random.random()
            if random.random() <= team_1_prob:
                self.teams[key] = team1id
            else:
                self.teams[key] = team2id

sample_submission = pd.read_csv("sample_submission.csv")
def run_simulation(gender):
    submission = sample_submission.copy()
    submission_gender = submission[submission['Tournament'] == gender]
    matchup_dict = {}
    w = Region("W",gender)
    x = Region("X",gender)
    y = Region("Y",gender)
    z = Region("Z",gender)

    ff = FinalFour(gender)
    for i in range(4):
        w.process_matchups(i+1)
        x.process_matchups(i+1)
        y.process_matchups(i+1)
        z.process_matchups(i+1)
    ff.teams["R4W1"] = w.champion
    ff.teams["R4X1"] = x.champion
    ff.teams["R4Y1"] = y.champion
    ff.teams["R4Z1"] = z.champion
    ff.process_matchups(5)
    ff.process_matchups(6)
    matchup_dict.update(w.teams)
    matchup_dict.update(x.teams)
    matchup_dict.update(y.teams)
    matchup_dict.update(z.teams)
    matchup_dict.update(ff.teams)
    submission_gender['Team'] = submission_gender['Slot'].map(matchup_dict)
    return submission_gender

def generate_portfolio(num_preds):
    sub = pd.DataFrame()
    for i in range(num_preds):
        mens_sub = run_simulation("M")
        womens_sub = run_simulation("W")
        mens_sub['Bracket'] = i+1
        womens_sub['Bracket'] = i+1
        sub = pd.concat([sub,mens_sub,womens_sub],ignore_index=True)
    sub = sub.reset_index()  # Reset the index to move it to a regular column
    sub.drop(columns=['RowId'], inplace=True,axis=0)  # Drop the existing 'RowID' column
    sub.rename(columns={'index': 'RowId'}, inplace=True)
    return sub


sub = generate_portfolio(100)
filtered_df = sub[(sub['Tournament'] == 'M') & (sub['Slot'] == 'R6CH')]
# Display unique values and their counts in the 'Team' column
team_counts = filtered_df['Team'].value_counts()
print(team_counts)