from matplotlib import pyplot as plt


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
