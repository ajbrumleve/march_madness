from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd

def getRatingStats(year,gender):
    url = f"https://www.sports-reference.com/cbb/seasons/{gender}/{year}-ratings.html"
    page = urlopen(url).read()
    soup = BeautifulSoup(page)
    count = 0
    table = soup.find("tbody")
    pre_df = dict()
    features_wanted = {'school_name', 'conf_abbr', 'wins', 'losses', 'pts_per_g', 'opp_pts_per_g', 'mov',
                       'sos', 'srs_off', 'srs_def', 'srs', 'off_rtg', 'def_rtg', 'net_rtg'}
    rows = table.find_all('tr')
    for row in rows:
        if (row.find('th', {"scope": "row"}) != None):
            for f in features_wanted:
                cell = row.find("td", {"data-stat": f})
                a = cell.text.strip().encode()
                text = a.decode("utf-8")
                if f in pre_df:
                    pre_df[f].append(text)
                else:
                    pre_df[f] = [text]
    df = pd.DataFrame.from_dict(pre_df)
    df["school_name"] = df["school_name"].apply(removeNCAA)
    df.to_csv(f"D:/PycharmProjects/march_madness/files/RatingStats/{gender}/RatingStats_{year}.csv")
    print(f"{year} finished")


def getMMStats(year,gender):
    url = f"https://www.sports-reference.com/cbb/seasons/{gender}/{year}-school-stats.html"
    page = urlopen(url).read()
    soup = BeautifulSoup(page)
    count = 0
    table = soup.find("tbody")
    pre_df = dict()
    features_wanted = {'school_name', 'g', 'wins', 'losses', 'win_loss_pct', 'srs', 'sos', 'wins_conf', 'losses_conf',
                       'wins_home', 'losses_home', 'wins_visitor', 'losses_visitor', 'pts', 'opp_pts', 'mp', 'fg',
                       'fga', 'fg_pct', 'fg3', 'fg3a', 'fg3_pct', 'ft', 'fta', 'ft_pct', 'orb', 'trb', 'ast', 'stl',
                       'blk', 'tov','pf'}
    rows = table.find_all('tr')
    for row in rows:
        if (row.find('th', {"scope": "row"}) != None):
            for f in features_wanted:
                cell = row.find("td", {"data-stat": f})
                a = cell.text.strip().encode()
                text = a.decode("utf-8")
                if f in pre_df:
                    pre_df[f].append(text)
                else:
                    pre_df[f] = [text]
    df = pd.DataFrame.from_dict(pre_df)
    df["school_name"] = df["school_name"].apply(removeNCAA)
    df.to_csv(f"D:/PycharmProjects/march_madness/files/MMStats/{gender}/MMStats_{year}.csv")
    print(f"{year} finished")

def removeNCAA(x):
    if("NCAA" in x):
        return x[:-5]
    else:
        return x

for year in range(1993,2024):
    getMMStats(year, "men")
    getRatingStats(year,"men")