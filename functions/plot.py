from matplotlib import pyplot as plt
import plotly.express as px


def plot_seperate(signal, reference):
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(signal,'blue',linewidth=1.5)
    ax2 = fig.add_subplot(212)
    ax2.plot(reference,'purple',linewidth=1.5)
    plt.show()


def plot_single(data, title='', x_lab='', y_lab=''):
    fig = px.line(data, title=title,
                  labels={
                      'index': x_lab,
                      'value': y_lab
                  })
    return fig


def heatmap(time_locked, mouse, event='FD'):
    fig = px.imshow(time_locked.T,
                    title='Evoked Responses for {mouse} for lever {event}'.format(mouse=mouse, event=event))
    fig.update_xaxes(side="top")
    fig.add_vline(x=360, line_dash="dash", line_color="green")
    return fig


def average_line(time_locked, mouse, ):
    fig = px.line(time_locked['average'], title='Average of {mouse}'.format(mouse=mouse),
                  labels={
                      'index': 'Time in Frames',
                      'value': 'zdFF'
                  })
    fig.add_vline(x=360, line_dash="dash", line_color="green")
    return fig
