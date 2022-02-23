from pathlib import Path

import pandas as pd
import plotly.express as px


DATA_DIR = Path(r'E:\independent test')
DATA_FILE = 'red_independence'
columns_r = ['Region0R', 'Region2R', 'Region4R', 'Region6R', 'Region8R', 'Region10R', 'Region12R']
columns_g = ['Region1G', 'Region3G', 'Region5G', 'Region7G', 'Region11G', 'Region13G', 'Region9G']

columns = columns_r

if __name__ == '__main__':
    df = pd.read_csv(DATA_DIR / DATA_FILE, delimiter=',')
    df = df[columns]
    df = df.diff()
    for column in columns:
        df[column] = df[df[column] > df[column].mean()][column]
    df = df.fillna(0)
    px.line(df).show()
