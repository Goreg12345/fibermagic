import os
import time
from functools import cache
from pathlib import Path

import dash
from dash import dcc, Input, Output
from dash import html
import plotly.express as px
import pandas as pd

from NeurophotometricsIO import reference, synchronize, perievents, create_giant_logs, create_giant_dataframe
from functions.perievents import single_perievent
from functions.plot import heatmap, average_line, plot_single, raster_plot

DATA_DIR = Path(r'C:\Users\Georg\OneDrive - UvA\0 Research\data\data_002')
DATA_FILE = 'FED3.csv'
FREQUENCY = 25
LOAD_FROM_DISC = True  # loading already saved tables from disc improves performance for e.g. debugging

overview = pd.read_csv(DATA_DIR / 'meta' / 'overview.csv', delimiter=';')
if not LOAD_FROM_DISC:
    logs = create_giant_logs(DATA_DIR)
    df = create_giant_dataframe(DATA_DIR, DATA_FILE)
    logs.to_csv('logs.csv')
    df.to_csv('data.csv')
else:
    logs = pd.read_csv('../logs.csv').set_index(['Analysis', 'Mouse'])
    df = pd.read_csv('../data.csv').set_index(['Analysis', 'Mouse', 'FrameCounter'])


def update_comparison():
    return html.Table()
    event = 'FD'
    paradigms = ['PR2', 'PR5', 'PR8']
    sensors = ['560', '470']
    table = html.Table(
        style={'width': '100%'},
        children=[
            html.Tr([html.Th(label, style={'width': '16%'}) for label in ['PR2, 560', 'PR2, 470', 'PR5, 560', 'PR5, 470', 'PR8, 560', 'PR8, 470']])
        ]
    )
    for mouse in overview.Mouse:
        row_table = html.Tr(children=[])
        for paradigm in paradigms:
            for sensor in sensors:
                sdf = df.loc[(paradigm, mouse), sensor]
                locked = perievents(sdf, logs.loc[(paradigm, mouse)].reset_index(), event, 8,
                                    frequency=FREQUENCY)
                row_table.children.append(html.Th(
                    dcc.Graph(
                        id='{m}-{p}-{s}'.format(m=mouse, p=paradigm, s=sensor),
                        figure=heatmap(locked, mouse, paradigm, sensor, event),
                        style={'display': 'inline-block', 'width': '100%'}
                    )
                ))
        table.children.append(row_table)
    return table


@cache
def get_data(data_file, analysis, region, wave_len, lever='FD'):
    sdf = pd.read_csv(DATA_DIR / analysis / DATA_FILE)
    if 'Flags' in sdf.columns:  # legacy fix: Flags were renamed to LedState
        sdf = sdf.rename(columns={'Flags': 'LedState'})
    sync_signals = pd.read_csv(DATA_DIR / analysis / 'input1.csv')
    timestamps = pd.read_csv(DATA_DIR / analysis / 'time.csv')
    logs = pd.read_csv(DATA_DIR / data_file)

    sdf = reference(sdf, region, wave_len)

    logs = synchronize(logs, sync_signals, timestamps)
    time_locked = single_perievent(sdf.zdFF, logs, 'FD', 15, FREQUENCY)
    fig = heatmap(time_locked, 'small mouse')
    fig2 = average_line(time_locked, 'small mouse')
    fig3 = plot_single(sdf[wave_len], 'Raw Data', 'Time in Frames', 'Absolute Value')
    fig4 = plot_single(sdf[410], 'Reference Data', 'Time in Frames', 'Absolute Value')
    fig5 = plot_single(sdf['zdFF'], 'Fitted Data', 'Time in Frames', 'zdFF')
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
                                       #value=['Rdlight'],
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
        ]),
        dcc.Tab(label='Comparison', id='comparison', children=[
            #dcc.Dropdown(
            #    id='mouse-dropdown',
            #    options=[{'label': mouse, 'value': mouse} for mouse in list(overview.Mouse)],
            #    #value=['MTL', 'SF'],
            #    multi=False
            #),
            html.Div(
                id='comparison-plot',
                children=update_comparison()
            )
        ])
    ])
])


#  @app.callback(
#      Output('comparison-plot', 'children'),
#      Input('mouse-dropdown', 'value')
#  )
#  def update_comparison(mouse):
#      if not mouse:
#          return []
#      event = 'FD'
#      children = []
#      for paradigm in ['PR2', 'PR5', 'PR8']:
#          for sensor in ['560', '470']:
#              sdf = df.loc[(paradigm, mouse), sensor]
#              locked = lock_time_to_event(sdf, logs.loc[(paradigm, mouse)].reset_index(), event, 15, frequency=FREQUENCY)
#              children.append(
#                  dcc.Graph(
#                      id='{p}-{s}'.format(p=paradigm, s=sensor),
#                      figure=heatmap(locked, mouse, event)
#                  )
#              )
#      return children


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
    wave_len = 560 if marker=='560' else 470
    print(wave_len, overview[overview.Mouse == mouse][marker])
    return get_data(analysis, paradigm, overview[overview.Mouse == mouse][marker].iloc[0], wave_len)


if __name__ == '__main__':
    app.run_server(debug=True)
