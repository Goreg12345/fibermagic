from scipy import signal as sgn
from tdt import read_block
import plotly.express as px
import numpy as np

from functions.martianova import smooth_signal


def main(block_paths):
    signals = []
    references = []
    for block in block_paths:
        data_struct = read_block(block)
        # use _470A for striatum and _470B for prefrontal cortex
        signals.append(data_struct.streams._470B.data)
        references.append(data_struct.streams._405B.data)

    # lower estimate: align data by adjusting by mean
    m1 = signals[0].mean()
    m2 = signals[1].mean()
    m3 = signals[2][0:200000].mean()
    s0 = signals[0] - (m1-m2)
    s2 = signals[2] - (m3-m2)
    lower_estimate = np.concatenate((s0, signals[1], s2))
    lower_estimate = (lower_estimate - np.median(lower_estimate)) / np.std(lower_estimate)
    lower_estimate = sgn.resample(lower_estimate, 90)  # recording is 90 minutes, we want one data point per minute

    signal = np.concatenate(signals)
    reference = np.concatenate(references)
    signal = sgn.resample(signal, 100000)
    reference = sgn.resample(reference, 100000)

    reference = smooth_signal(reference, 10)
    signal = smooth_signal(signal, 10)

    reference = (reference - np.median(reference)) / np.std(reference)
    signal = (signal - np.median(signal)) / np.std(signal)

    zdFF = signal - reference
    higher_estimate = sgn.resample(zdFF, 90)
    higher_estimate -= higher_estimate[0:25].mean() - lower_estimate[0:25].mean()

    px.line(lower_estimate).add_scatter(y=higher_estimate, mode='lines').show()


if __name__ == '__main__':
    main([
        r'C:\Users\Georg\OneDrive - UvA\0 Research\data_amphetamine\Data\20210613\m919_10mgkg_amph\m919_10mgkg_amph-210613-162902',
        r'C:\Users\Georg\OneDrive - UvA\0 Research\data_amphetamine\Data\20210613\m919_10mgkg_amph\m919_10mgkg_amph-210613-164542',
        r'C:\Users\Georg\OneDrive - UvA\0 Research\data_amphetamine\Data\20210613\m919_10mgkg_amph\m919_10mgkg_amph-210613-170136'
    ]
    )
