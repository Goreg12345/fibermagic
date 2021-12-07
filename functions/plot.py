from matplotlib import pyplot as plt
import plotly.express as px


def plot_seperate(signal, reference):
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(signal,'blue',linewidth=1.5)
    ax2 = fig.add_subplot(212)
    ax2.plot(reference,'purple',linewidth=1.5)
    plt.show()


def plot_single(data):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(data, 'black', linewidth=1.5)
    plt.show()


def heatmap(time_locked):
    fig = px.imshow(time_locked.T)
    fig.update_xaxes(side="top")
    fig.show()
