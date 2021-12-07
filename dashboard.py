from pathlib import Path

import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd

from NeurophotometricsIO import reference, synchronize, lock_time_to_event
from functions.plot import heatmap, average_line

DATA_DIR = Path(r'C:\Users\Georg\OneDrive - UvA\0 Research\data')
ANALYSIS = 'PR5'
DATA_FILE = 'FED3.csv'
LOG_FILE = 'FRfD2AdoraCrepilotPR5B8624-1.log'
MOUSE_NAME = 'B8624'
LED = '7'
FREQUENCY = 25

df = pd.read_csv(DATA_DIR / ANALYSIS / DATA_FILE)
df = reference(df, 'Region6R', 560)

sync_signals = pd.read_csv(DATA_DIR / ANALYSIS / 'input1.csv')
timestamps = pd.read_csv(DATA_DIR / ANALYSIS / 'time.csv')
logs = pd.read_csv(DATA_DIR / ANALYSIS / LOG_FILE)
logs = synchronize(logs, sync_signals, timestamps)

time_locked = lock_time_to_event(df, logs, 'FD', 15 * FREQUENCY)
fig = heatmap(time_locked, 'small mouse')
fig2 = average_line(time_locked, 'small mouse')
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Fiber Magic'),

    html.Div(children='''
        Pilot Experiment.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    dcc.Graph(
        id='example-graph-2',
        figure=fig2
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)