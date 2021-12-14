import os
import time
from functools import cache
from pathlib import Path

import dash
from dash import dcc, Input, Output
from dash import html
import plotly.express as px
import pandas as pd

from NeurophotometricsIO import reference, synchronize, lock_time_to_event, create_giant_logs, create_giant_dataframe
from functions.plot import heatmap, average_line, plot_single, raster_plot

DATA_DIR = Path(r'C:\Users\Georg\OneDrive - UvA\0 Research\data_001')
DATA_FILE = 'FED3.csv'
FREQUENCY = 25

overview = pd.read_csv(DATA_DIR / 'meta' / 'overview.csv', delimiter=';')
#logs = create_giant_logs(DATA_DIR)
#df = create_giant_dataframe(DATA_DIR, DATA_FILE)
#logs.to_csv('logs.csv')
#df.to_csv('data.csv')
logs = pd.read_csv('logs.csv').set_index(['Analysis', 'Mouse'])
df = pd.read_csv('data.csv').set_index(['Analysis', 'Mouse'])


@cache
def get_data(data_file, analysis, region, wave_len, lever='FD'):
    t1 = time.time()
    df = pd.read_csv(DATA_DIR / analysis / DATA_FILE)
    sync_signals = pd.read_csv(DATA_DIR / analysis / 'input1.csv')
    timestamps = pd.read_csv(DATA_DIR / analysis / 'time.csv')
    logs = pd.read_csv(DATA_DIR / data_file)

    df = reference(df, region, wave_len)

    logs = synchronize(logs, sync_signals, timestamps)
    time_locked = lock_time_to_event(df, logs, lever, 15 * FREQUENCY)

    fig = heatmap(time_locked, 'small mouse')
    fig2 = average_line(time_locked, 'small mouse')
    fig3 = plot_single(df[wave_len], 'Raw Data', 'Time in Frames', 'Absolute Value')
    fig4 = plot_single(df[410], 'Reference Data', 'Time in Frames', 'Absolute Value')
    fig5 = plot_single(df['zdFF'], 'Fitted Data', 'Time in Frames', 'zdFF')
    return fig, fig2, fig3, fig4, fig5


#########################################################
def list_trials_and_mice():
    analyses = list()
    for analysis in os.listdir(DATA_DIR):
        for file in os.listdir(DATA_DIR / analysis):
            if '.log' in file:
                analyses.append(
                    {'label': 'Paradigm: {p}, Mouse: {m}'.format(p=analysis, m=file), 'value': str(Path(analysis) / file)}
                )
    return analyses
#########################################################
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Fiber Photometry'),

    dcc.Tabs(id='main_tab', value='raw-data', children=[
        dcc.Tab(label='Raw Data', id='raw-data', children=[
            html.Div(children=[
                html.Div(children=['Pilot Experiment.',
                                   dcc.Dropdown(
                                       id='analysis-dropdown',
                                       options=list_trials_and_mice(),
                                       #value=['MTL', 'SF'],
                                       multi=False
                                   ),
                                   dcc.Dropdown(
                                       id='marker',
                                       options=[{'label': 'Gcamp', 'value': '470'},
                                                {'label': 'Rdlight', 'value': '560'}],
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
        ]),
        dcc.Tab(label='Events and Statistics', id='events', children=[
            dcc.Dropdown(
                id='analysis-dropdown-events',
                options=list_trials_and_mice(),
                #value=['MTL', 'SF'],
                multi=False
            ),
            dcc.Graph(
                id='logs-raster',
                figure=px.scatter(pd.DataFrame())
            )
        ])
    ])
])


@app.callback(
    Output('logs-raster', 'figure'),
    Input('analysis-dropdown-events', 'value')
)
def update_raster(analysis):
    if not analysis:
        return px.scatter(pd.DataFrame())
    paradigm = analysis.split('\\')[0]
    mouse = analysis.split('-1.log')[0][-5:]
    return raster_plot(logs.loc[(paradigm, mouse)], events=['LL', 'FD'])


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
    if not analysis or not marker:
        return [px.line(pd.DataFrame())] * 5
    paradigm = analysis.split('\\')[0]
    mouse = analysis.split('-1.log')[0][-5:]
    wave_len = 560 if marker=='Rdlight' else 470
    return get_data(analysis, paradigm, overview[overview.Mouse == mouse][marker].iloc[0], wave_len)


if __name__ == '__main__':
    app.run_server(debug=True)
