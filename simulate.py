import pandas as pd
import statistics
team_names = pd.read_csv('files/MTeams.csv')

def get_pred(team1, team2, matchups_df):
    """
    Retrieves the win probability for team1 vs team2.
    """
    row = matchups_df[
        ((matchups_df["team1"] == team1) & (matchups_df["team2"] == team2)) |
        ((matchups_df["team1"] == team2) & (matchups_df["team2"] == team1))
        ]

    if row.empty:
        return 0.5  # Default to 50-50 if missing
    elif row.iloc[0]["team1"] == team1:
        return row.iloc[0]["win_probability"]
    else:
        return 1 - row.iloc[0]["win_probability"]


def generate_possible_matchups(remaining_teams, round_num, seed_matchups):
    """
    Generate all possible matchups for the current round based on seedings.
    """
    matchups = []
    num_teams = len(remaining_teams)

    # Iterate over the remaining teams and their possible matchups based on seedings
    for i in range(0, num_teams, 2):
        team1 = remaining_teams[i]
        team2 = remaining_teams[i + 1]

        # Get the seed for each team (assuming teams are sorted by seed)
        seed1 = team1['seed']
        seed2 = team2['seed']

        # Find possible matchups based on seed matchups for the round
        if round_num == 1:
            matchups.append((team1, team2))  # Round 1 uses the fixed bracket
        else:
            possible_opponents_1 = seed_matchups[round_num].get(seed1, [])
            possible_opponents_2 = seed_matchups[round_num].get(seed2, [])

            # Generate all possible matchups from seed matchups for round 2+ onwards
            matchups.extend([(team1, opponent) for opponent in possible_opponents_1])
            matchups.extend([(team2, opponent) for opponent in possible_opponents_2])

    return matchups


def simulate_region(region_name, region_df, matchups_df, seed_matchups, min_prob_threshold=0.0001):
    """
    Simulates the region, keeping track of all possible matchups in each round.
    """
    round_num = 1
    region_probs = {team: {1: 1.0} for team in region_df["team"]}  # Start with probability 1.0 for Round 1
    remaining_teams = region_df["team"].tolist()

    # Initialize probabilities for all rounds
    max_rounds = 5  # Adjusting to account for possible rounds beyond 4
    for team in region_probs:
        for r in range(2, max_rounds + 1):
            region_probs[team][r] = 0.0  # Set initial probability for all rounds to 0 for future rounds

    # Iterate through the rounds
    while len(remaining_teams) > 1:
        print(f"\n=== {region_name} - ROUND {round_num} ===")

        # Generate all possible matchups based on seed matchups for the round
        matchups = generate_possible_matchups(remaining_teams, round_num, seed_matchups)
        new_probs = {}

        # Track teams for the next round
        next_round_teams = []

        for team1, team2 in matchups:
            prob_t1_win, prob_t2_win = simulate_matchup(team1, team2, matchups_df)

            # Multiply previous round's probability with the win probability
            prob_t1 = region_probs[team1][round_num] * prob_t1_win
            prob_t2 = region_probs[team2][round_num] * prob_t2_win

            # Apply threshold: if probability is below threshold, still track it but don't eliminate
            prob_t1 = max(prob_t1, min_prob_threshold)
            prob_t2 = max(prob_t2, min_prob_threshold)

            # Save updated probabilities for both teams
            new_probs[team1] = prob_t1
            new_probs[team2] = prob_t2

            # Add teams with any probability to the next round
            if prob_t1 > min_prob_threshold:
                next_round_teams.append(team1)
            if prob_t2 > min_prob_threshold:
                next_round_teams.append(team2)

            print(f"{team1} ({prob_t1:.2%}) vs {team2} ({prob_t2:.2%})")

        # Update remaining teams to include only those that have a probability above the threshold
        remaining_teams = next_round_teams

        # Store results for the next round
        round_num += 1
        for team in new_probs:
            region_probs[team][round_num] = new_probs[team]

    return remaining_teams[0], region_probs


def simulate_full_tournament(teams_df, matchups_df, seed_matchups):
    """
    Simulates the entire tournament with 4 regions, taking into account the multiple possible matchups in each round.
    """
    regions = ["W", "E", "S", "M"]  # Example: West, East, South, Midwest
    region_results = {}

    for region in regions:
        print(f"\n=== {region} Region Simulation ===")
        region_df = teams_df[teams_df["region"] == region]
        region_results[region] = simulate_region(region, region_df, matchups_df, seed_matchups)

    return region_results

def parse_submission_df():
    df = pd.read_csv('2025_saved_data/submission.csv')
    df[['team1', 'team2']] = df['ID'].apply(lambda x: pd.Series(x.split("_")[1:3]))
    df.rename(columns={'Pred': 'win_probability'}, inplace=True)
    df = df.drop(['ID'], axis=1)
    df['team1'] = df['team1'].astype(int)
    df['team2'] = df['team2'].astype(int)

    return df


def parse_team_seeds():
    df = pd.read_csv('files/MNCAATourneySeeds.csv')
    df = df[df['Season'] == 2025]

    df['region'] = df['Seed'].str[0]  # Extract first character
    df['seed'] = df['Seed'].str[1:3]
    df.rename(columns={'TeamID': 'team'}, inplace=True)
    df['team'] = df['team'].astype(int)
    df = df.drop(['Season', 'Seed'], axis=1)
    return df


teams_df = parse_team_seeds()
matchups_df = parse_submission_df()

# Mark duplicates within each region and seed, keeping only the first occurrence
teams_df = teams_df[~teams_df.duplicated(subset=['region', 'seed'], keep='first')]


seed_matchups = {1:{1:[16],2:[15],3:[14],4:[13],5:[12],6:[11],7:[10],8:[9]},
 2:{1:[8,9],16:[8,9],2:[7,10],15:[7,10], 3:[6,11], 14:[6,11], 4:[5,12], 13:[5,12]},
 3:{1:[4,5,12,13],8:[4,5,12,13], 9:[4,5,12,13], 16: [4,5,12,13], 2:[3,6,11,14], 7:[3,6,11,14],10:[3,6,11,14], 15:[3,6,11,14]},
 4:{1:[2,3,6,7,10,11,14,15],4:[2,3,6,7,10,11,14,15],5:[2,3,6,7,10,11,14,15],8:[2,3,6,7,10,11,14,15],9:[2,3,6,7,10,11,14,15],12:[2,3,6,7,10,11,14,15],13:[2,3,6,7,10,11,14,15],16:[2,3,6,7,10,11,14,15]}
 }


def getTeamID(team, region):
    return teams_df[(teams_df['seed'].astype(int) == team) & (teams_df['region'] == region)]['team'].iloc[0]

def getTeamName(teamid):
    return team_names[team_names['TeamID'].astype(int) == teamid]['TeamName'].iloc[0]
region_preds = {}
for region in ["W","X","Y","Z"]:
    print(f"in Region {region}:")
    preds = {}
    preds[0] = {}
    for round in seed_matchups.keys():
        for team1,team2_arr in seed_matchups[round].items():

            for team2 in team2_arr:
                team1ID = getTeamID(team1,region)
                team2ID = getTeamID(team2,region)
                team1pred = get_pred(team1ID, team2ID, matchups_df)
                team2pred = 1-team1pred
                preds[round-1][team1ID] = preds[round-1].get(team1ID,1)
                preds[round-1][team2ID] = preds[round-1].get(team2ID,1)
                preds[round] = preds.get(round,{})
                preds[round][team1ID] = preds[round].get(team1ID,[])
                sit_pred = preds[round-1][team1ID]*preds[round-1][team2ID]
                preds[round][team1ID].append(team1pred*sit_pred)
                preds[round][team2ID] = preds[round].get(team2ID, [])
                preds[round][team2ID].append(team2pred*sit_pred)
        for team in preds[round].keys():
            preds[round][team] = sum(preds[round][team])
        print(f"Round {round}")
        sorted_preds = sorted(preds[round].items(), key=lambda x: x[1], reverse=True)
        # Assuming getTeamName is a function that takes a key and returns the team name
        for key, value in sorted_preds:
            print(f"{getTeamName(key)} - {value * 100:.2f}%")
        print("\n")



    print("\n\n")
    region_preds[region] = preds
                
# Simulate full tournament
# region_results = simulate_full_tournament(teams_df, matchups_df, seed_matchups)
#
# # Print cumulative probabilities for each team
# for region, teams in region_results.items():
#     print(f"\n=== {region} Region Cumulative Probabilities ===")
#     for team, rounds in teams.items():
#         print(f"{team}: " + " | ".join([f"Round {r}: {p:.4%}" for r, p in rounds.items()]))
