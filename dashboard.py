import os
from pathlib import Path

import dash
from dash import dcc, Input, Output
from dash import html
import plotly.express as px
import pandas as pd

from NeurophotometricsIO import reference, synchronize, lock_time_to_event
from functions.plot import heatmap, average_line, plot_single

DATA_DIR = Path(r'C:\Users\Georg\OneDrive - UvA\0 Research\data')
ANALYSIS = 'PR5'
DATA_FILE = 'FED3.csv'
LOG_FILE = 'FRfD2AdoraCrepilotPR5B8624-1.log'
MOUSE_NAME = 'B8624'
LED = '7'
FREQUENCY = 25

overview = pd.read_csv(DATA_DIR / 'meta' / 'overview.csv', delimiter=';')
def get_figures(data_file, analysis, region, wave_len, lever='FD'):
    df = pd.read_csv(DATA_DIR / analysis / DATA_FILE)
    df = reference(df, region, wave_len)

    sync_signals = pd.read_csv(DATA_DIR / analysis / 'input1.csv')
    timestamps = pd.read_csv(DATA_DIR / analysis / 'time.csv')
    logs = pd.read_csv(DATA_DIR / data_file)
    logs = synchronize(logs, sync_signals, timestamps)

    time_locked = lock_time_to_event(df, logs, lever, 15 * FREQUENCY)
    fig = heatmap(time_locked, 'small mouse')
    fig2 = average_line(time_locked, 'small mouse')
    fig3 = plot_single(df[wave_len], 'Raw Data', 'Time in Frames', 'Absolute Value')
    fig4 = plot_single(df[410], 'Reference Data', 'Time in Frames', 'Absolute Value')
    fig5 = plot_single(df['zdFF'], 'Fitted Data', 'Time in Frames', 'zdFF')
    return fig, fig2, fig3, fig4, fig5

#########################################################
analyses = list()
for analysis in os.listdir(DATA_DIR):
    for file in os.listdir(DATA_DIR / analysis):
        if '.log' in file:
            analyses.append(
                {'label': 'Paradigm: {p}, Mouse: {m}'.format(p=analysis, m=file), 'value': str(Path(analysis) / file)}
            )
#def get_file_names(analysis):
#    files = []
#    for file in os.listdir(DATA_DIR / analysis):
#        if '.log' in file:
#            files.append(file)
#    return [{'label': i, 'value': i} for i in files]
#########################################################
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Fiber Magic'),

    html.Div(children=['''
        Pilot Experiment.
    ''',
       dcc.Dropdown(
           id='analysis-dropdown',
           options=analyses,
           #value=['MTL', 'SF'],
           multi=False
       ),
       dcc.Dropdown(
           id='marker',
           options=[{'label': 'Gcamp', 'value': 'Gcamp'},
                    {'label': 'Rdlight', 'value': 'Rdlight'}],
           value=['Rdlight'],
           multi=False
       ),
    ]),

    dcc.Graph(
        id='example-graph',
        figure=px.line(pd.DataFrame())
    ),
    dcc.Graph(
        id='example-graph-2',
        figure=px.line(pd.DataFrame())
    ),
    dcc.Graph(
        id='example-graph-3',
        figure=px.line(pd.DataFrame())
    ),
    dcc.Graph(
        id='example-graph-4',
        figure=px.line(pd.DataFrame())
    ),
    dcc.Graph(
        id='example-graph-5',
        figure=px.line(pd.DataFrame())
    )
])


@app.callback(
    Output('example-graph', 'figure'),
    Output('example-graph-2', 'figure'),
    Output('example-graph-3', 'figure'),
    Output('example-graph-4', 'figure'),
    Output('example-graph-5', 'figure'),
    Input('analysis-dropdown', 'value'),
    Input('marker', 'value')
)
def update_graph(analysis, marker):
    if not analysis:
        return [px.line(pd.DataFrame())] * 4
    print(analysis)
    paradigm = analysis.split('\\')[0]
    mouse = analysis.split('-1.log')[0][-5:]
    print(mouse, overview[overview.Mouse==mouse])
    return get_figures(analysis, paradigm, overview[overview.Mouse==mouse][marker].iloc[0],
                       560 if marker=='Rdlight' else 470)


if __name__ == '__main__':
    app.run_server(debug=True)