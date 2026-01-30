# library imports
import time

# linear algebra
import numpy as np

## file IO / processing
import pandas as pd

## system
import os

## requests
import requests

## results storage
from collections import defaultdict

from bs4 import BeautifulSoup


## functions

def get_soup(url, parser='html.parser', verify=True):
    """
        Gets the Beautifulsoup object which holds the data to be scrapped

        Args:
            url (str): URL of the page to be scrapped
            parser (str): Type of parser to be used
        Returns:
            BeautifulSoup object


    """
    request = get_request_response(url, verify=verify)

    return BeautifulSoup(request.content, parser)


def get_request_response(url, verify=True):
    """

        Requests the content to be scrapped

        Args:
            url (url): URL of the page to be scrapped

        Returns:
            request object

    """

    return requests.get(url, verify=verify)


def find_all(soup, **kwargs):
    """
        Returns all the PageElement tags which match a given criteria
        See BeautifulSoup find_all documentation

        Args:
            soup (BeautifulSoup): BeautifulSoup object containing the web page content

        Returns:
            list of PageElements


    """
    return soup.find_all(**kwargs)


def find(soup, **kwargs):
    """
        Returns one PageElement tag which matches a given criteria
        See BeautifulSoup find_all documentation

        Args:
            soup (BeautifulSoup): BeautifulSoup object containing the web page content

        Returns:
            PageElement instance



    """
    return soup.find(**kwargs)


def find_per_game_table(soup):
    """

        Returns per game stats table PageElement tag

        See BeautifulSoup find_all documentation

        Args:
            soup (BeautifulSoup): BeautifulSoup object containing the web page content

        Returns:
            per game stats table PageElement tag instance



    """
    return find(soup, name='table', attrs={'id': 'players_per_game'})


def find_stats_tags_from_soup(soup, name):
    """

        Returns table PageElement tag of the chosen table

        See BeautifulSoup find_all documentation

        Args:
            soup (BeautifulSoup): BeautifulSoup object containing the web page content
            name (str): name of the table to be scraped

        Returns:
            table PageElement tag instance



    """
    if name == 'per game':
        table_soup = find_per_game_table(soup)
    else:
        raise NotImplementedError('Invalid table name')
    if table_soup:
        pass
    return find_all(soup=table_soup, name='td')


def get_stats(tags):
    """
        Get stats from the table tags

        Args:
            tags (list(PageElement tag)): list of tag elements which contain player informations

        returns:
            dictionary of player stats


    """
    game_stats = defaultdict(list)

    for tag in tags:
        game_stats[tag['data-stat']].append(tag.string)

    return game_stats


def get_stats_from_soup(soup, name):
    """
        Get player stats from BeautifulSoup object

        Args:
            soup (BeautifulSoup): BeautifulSoup object containing the web page content
            name (str): name of the table to be scraped

        Returns:
            dictionary of player stats


    """
    tags = find_stats_tags_from_soup(soup, name=name)
    return get_stats(tags)


def get_stats_frame_from_soup(soup, name):
    """
        Get player stats DataFrame from BeautifulSoup object

        Args:
            soup (BeautifulSoup): BeautifulSoup object containing the web page content
            name (str): name of the table to be scraped

        Returns:
            pandas DataFrame of player stats


    """
    return pd.DataFrame(data=get_stats_from_soup(soup, name=name))


def scrape_per_game_stats_from_url(url):
    """
        Get player stats from BeautifulSoup object with school and season

        Args:
            soup (BeautifulSoup): BeautifulSoup object containing the web page content
            name (str): name of the table to be scraped

        Returns:
            pandas DataFrame of player stats


    """
    soup = get_soup(url=url)
    df = get_stats_frame_from_soup(soup=soup, name='per game')
    df['ast_per_g'] = pd.to_numeric(df['ast_per_g'], errors='coerce')
    df['fg_per_g'] = pd.to_numeric(df['fg_per_g'], errors='coerce')
    df['mp_per_g'] = pd.to_numeric(df['mp_per_g'], errors='coerce')
    df['fg3_pct'] = pd.to_numeric(df['fg3_pct'], errors='coerce')
    df['fg3a_per_g'] = pd.to_numeric(df['fg3a_per_g'], errors='coerce')
    df['tov_per_g'] = pd.to_numeric(df['tov_per_g'], errors='coerce')
    qual_guard_count = len(df[(df['pos'] == 'G') & (df['ast_per_g'] > df['tov_per_g']) & (df['mp_per_g'] > 20)])
    qual_forward_count = len(df[(df['pos'] == 'F') & (df['fg3_pct'] > .35) & (df['mp_per_g'] > 20) & (df['fg3a_per_g'] > 1)])
    return df, qual_guard_count, qual_forward_count
def validate_spelling(gender, teams_pd):
    # Load team spellings
    team_spellings = pd.read_csv(
        r'C:/Users/abrumleve4982/PyCharm Projects/march_madness/files/MTeamSpellings.csv',
        encoding='cp1252'
    )

    # Iterate through each team spelling
    for index, row in team_spellings.iterrows():
        time.sleep(4)
        team = row['TeamNameSpelling']
        print(f"Checking: {team}")

        # Build URL for team page
        url = f"https://www.sports-reference.com/cbb/schools/{team}/{gender}/2021.html#all_schools_per_game"

        try:
            response = requests.get(url)
            if response.status_code == 200:  # Valid spelling
                print(f"Valid: {team}")

                # Update the 'sr_spelling' column for matching TeamID in teams_pd
                teams_pd.loc[teams_pd['TeamID'] == row['TeamID'], 'sr_spelling'] = team
            elif response.status_code == 429:
                print(f"Rate limit met: {team})")
                break
        except Exception as e:
            print(f"Error checking {team}: {e}")
            continue  # Skip to the next team

    return teams_pd  # Return updated DataFrame
    # for team in teams_pd['TeamNa']
def build_player_quality_df(gender):
    # Load team spellings
    if gender == 'men':
        gender_code = 'M'
    else:
        gender_code = "W"
    team_spellings = pd.read_csv(
        f'C:/Users/abrumleve4982/PyCharm Projects/march_madness/files/{gender_code}TeamSpellings.csv',
        encoding='cp1252'
    )


    try:
        player_quality_df = pd.read_csv(f'{gender_code}PlayerQuality2.csv')
        last_team_id = player_quality_df.iloc[-1]['TeamID']
        index_to_cut = team_spellings[team_spellings['TeamID'] == last_team_id].index.min()
        # Remove that row and all previous ones
        if not pd.isna(index_to_cut):
            team_spellings = team_spellings.loc[index_to_cut + 1:]
    except:
        player_quality_df = pd.DataFrame(columns=['TeamID','Gender','Season','Quality_G_Count','Quality_F_Count'])
    # Iterate through each team spelling

    for index, row in team_spellings.iterrows():
        time.sleep(3.1)
        team = row['TeamNameSpelling']
        print(f"Checking: {team}")
        year = 2024
        # Build URL for team page
        url = f"https://www.sports-reference.com/cbb/schools/{team}/{gender}/{year}.html#all_schools_per_game"

        try:
            response = requests.get(url)
            if response.status_code == 200:  # Valid spelling
                print(f"Valid: {team}")
                for year in range(1990,2026):
                    time.sleep(3.1)
                    try:
                        url = f"https://www.sports-reference.com/cbb/schools/{team}/{gender}/{year}.html#all_schools_per_game"
                        df, qual_guard_count, qual_forward_count = scrape_per_game_stats_from_url(url)
                        new_row = {'TeamID':row['TeamID'],'Gender':gender, 'Season':year,'Quality_G_Count':qual_guard_count,'Quality_F_Count':qual_forward_count}
                        player_quality_df = pd.concat([player_quality_df, pd.DataFrame([new_row])], ignore_index=True)
                    except:
                        continue
                player_quality_df.to_csv(f'{gender_code}PlayerQuality2.csv',index=False)
            elif response.status_code == 429:
                print(f"Rate limit met: {team})")
                break
        except Exception as e:
            print(f"Error checking {team}: {e}")
            continue  # Skip to the next team
    # player_quality_df.to_csv(f'{gender_code}PlayerQuality.csv',index=False)
    return player_quality_df  # Return updated DataFrame

# url = "https://www.sports-reference.com/cbb/schools/gonzaga/2021.html#all_schools_per_game"
# df = scrape_per_game_stats_from_url(url=url, school='Gonzaga', season='2020-2021')
# df['ast_per_g'] = pd.to_numeric(df['ast_per_g'], errors='coerce')
# df['fg_per_g'] = pd.to_numeric(df['fg_per_g'], errors='coerce')
# df['mp_per_g'] = pd.to_numeric(df['mp_per_g'], errors='coerce')
# df['fg3_pct'] = pd.to_numeric(df['mp_per_g'], errors='coerce')
# df['tov_per_g'] = pd.to_numeric(df['mp_per_g'], errors='coerce')
# len(df[(df['pos'] == 'G') & (df['ast_per_g'] > df['tov_per_g']) & (df['mp_per_g'] > 10)])
# len(df[(df['pos'] == 'F') & (df['fg3_pct'] > 35)])
build_player_quality_df('men')
build_player_quality_df('women')