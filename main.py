import pandas as pd

# Read the files
df1 = pd.read_csv('England CSV.csv')
df2 = pd.read_csv('England 2 CSV.csv')


# pd.set_option('display.max_columns', None) <<<Show all columns>>>
# print(df1.head())
# print(df2.head())

def get_match_result(row):
    if row['FTH Goals'] > row['FTA Goals']:
        return 'H'  # Home Win
    elif row['FTH Goals'] < row['FTA Goals']:
        return 'A'  # Away Win
    else:
        return 'D'  # Draw


df1['Match_Result'] = df1.apply(get_match_result, axis=1)
df2['Match_Result'] = df2.apply(get_match_result, axis=1)

columns_to_keep = [
    'HomeTeam', 'AwayTeam',
    'H Shots', 'A Shots',
    'H SOT', 'A SOT',
    'H Corners', 'A Corners',
    'H Fouls', 'A Fouls',
    'H Yellow', 'A Yellow',
    'H Red', 'A Red',
    'League',
    'Match_Result'
]

df1 = df1[columns_to_keep]
df2 = df2[columns_to_keep]
