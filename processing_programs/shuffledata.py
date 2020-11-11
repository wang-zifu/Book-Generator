import pandas as pd
from sklearn.utils import shuffle

df = pd.read_json('./gutenberg.json', orient='records', lines=True)
df = shuffle(df)
df.to_json('./shuffledgutenberg.json', orient='records', lines=True)