import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

from tqdm.auto import tqdm

tqdm.pandas()


def demodulate(
    data=None, timestamps=None, signal=None, isosbestic=None, by=None, steps=False, method="airPLS", **kwargs
):
    """
    High-level function to demodulate signal

    Parameters
    ----------
    data : pd.DataFrame or dict with raw data
        the data to demodulate, if None, timestamps, signal and isosbestic must be provided as array-like
    timestamps : str or array-like of the column with timestamps; if str, it is the name of the column in data
        the column with timestamps or the timestamps themselves
    signal : str or array-like of the column with signal; if str, it is the name of the column in data
        the column with signal or the signal itself
    isosbestic : str or array-like of the column with isosbestic signal; if str, it is the name of the column
    in data the column with isosbestic signal or the isosbestic signal itself
    by : str or list with column names to group by or array-like with group labels, has to be the same length as
        the data
        divides the data into groups and demodulates each group separately, typically you should pass enough
        columns to uniquely identify each session
    steps : bool
        if True, returns intermediate steps as a pd.DataFrame
    method : str
        method to use for demodulation, one of [airPLS, biexponential decay] or None if only artifact removal
    kwargs : dict
        parameters for airPLS or biexponential decay.

        For airPLS:
        lambda: parameter for airPLS; The larger lambda is, the smoother the resulting background
        porder: parameter for airPLS; adaptive iteratively reweighted penalized least squares for baseline fitting
        itermax: parameter for airPLS; maximum number of iterations
        lambda_artifact: parameter for artifact removal; The larger lambda is, the smoother the resulting
        background
        porder_artifact: parameter for artifact removal; adaptive iteratively reweighted penalized least squares
        for
        baseline fitting
        itermax_artifact: parameter for artifact removal; maximum number of iterations

    Returns
    -------
    pd.DataFrame or pd.Series with the demodulated signal

    """

    demodulator = _Demodulator(
        data=data,
        timestamps=timestamps,
        signal=signal,
        isosbestic=isosbestic,
        by=by,
        steps=steps,
        method=method,
        **kwargs
    )

    return demodulator()


class _Demodulator:
    def __init__(
        self,
        data=None,
        timestamps=None,
        signal=None,
        isosbestic=None,
        by=None,
        steps=False,
        method="airPLS",
        **kwargs
    ):

        self.timestamp = timestamps
        self.signal = signal
        self.isosbestic = isosbestic

        # convert input to a common representation
        if type(data) != pd.DataFrame:
            if data:
                timestamps = data.get(timestamps, timestamps)
                signal = data.get(signal, signal)
                isosbestic = data.get(isosbestic, isosbestic)
            if any([type(x) == str for x in [timestamps, signal, isosbestic]]):
                raise ValueError("If data is not provided, timestamps, signal and isosbestic must be array-like")
            data = pd.DataFrame({"timestamps": timestamps, "signal": signal, "isosbestic": isosbestic})
        else:
            # check if column names is exists if they are not None
            if type(timestamps) == str and timestamps not in data.columns:
                raise ValueError("timestamps column not found in data")
            else:
                data = data.rename({timestamps: "timestamps"}, axis=1)
            if type(signal) == str and signal not in data.columns:
                raise ValueError("signal column not found in data")
            else:
                data = data.rename({signal: "signal"}, axis=1)
            if type(isosbestic) == str and isosbestic not in data.columns:
                raise ValueError("isosbestic column not found in data")
            else:
                data = data.rename({isosbestic: "isosbestic"}, axis=1)
        # check if by is a string or list of strings, all of which are in data.columns
        if by:
            if type(by) == str:
                if by not in data.columns:
                    raise ValueError("per column not found in data")
            elif type(by) == list:
                if any([x not in data.columns and x not in data.index.names for x in by]):
                    raise ValueError("per column not found in data")
            else:
                by = np.asarray(by)
                # check if len of per is equal to len of data
                if by and len(by) != len(data):
                    raise ValueError("per must be the same length as data")

        self.steps = steps
        self.method = method
        self.data = data

        self.by = by
        if by:
            # count number of unique groups in by columns
            # if only one group, set by to None to not groupby
            n_groups = len(data.groupby(by=by).groups)
            if n_groups == 1:
                self.by = None

        # set parameters
        self.lambda_ = kwargs.get("lambda", 5e4)
        self.porder = kwargs.get("porder", 1)
        self.itermax = kwargs.get("itermax", 50)

        self.lambda_artifact = kwargs.get("lambda_artifact", 1e4)
        self.porder_artifact = kwargs.get("porder_artifact", 1)
        self.itermax_artifact = kwargs.get("itermax_artifact", 50)

    def __call__(self):
        # select the correct method given the input and run it

        # if method None, only artifact removal
        if not self.method:
            if not self.data.isosbestic:
                raise ValueError("isosbestic must be provided for artifact removal")
            if not self.by:
                method = self.remove_artifact

        # if no isosbestic is provided, only demodulate, no artifact removal
        if not self.isosbestic:
            if self.method == "airPLS":
                method = self.demodulate_with_airPLS
            elif self.method == "biexponential decay":
                method = self.demodulate_with_biexponential_decay
        # if isosbestic is provided, demodulate and remove artifact
        else:
            if self.method == "airPLS":
                method = self.demodulate_with_airPLS_and_remove_artifact
            elif self.method == "biexponential decay":
                method = self.demodulate_with_biexponential_decay_and_remove_artifact

        if not self.by:
            return method(self.data)
        # don't sort and return index, because it should be possible to assign the result to the original index
        r = self.data.groupby(self.by, sort=False, as_index=False).progress_apply(method)
        if isinstance(r, pd.DataFrame) and len(r.columns.names) == 1:
            column_name = r.columns.names[0]
            r = r.stack()  # because progress_apply returns a df with one column not a series
            r.name = column_name
        return r.reset_index(level=0, drop=True)

    def demodulate_with_airPLS_and_remove_artifact(self):
        """
        Demodulate signal using airPLS and remove artifact, operates on numpy arrays
        Returns
        -------
        np.array with artifact removed signal or df with intermediate steps as additional columns

        """

        signal = self.demodulate_with_airPLS()
        if self.steps:
            # make dataframe with intermediate steps
            res_df = signal
            signal = signal["dFF"].to_numpy()
        signal = self.remove_artifact(signal)
        if self.steps:
            return pd.concat([res_df, signal], axis=1)
        return signal

    def demodulate_with_biexponential_decay_and_remove_artifact(timestamps, signal, isosbestic, steps=False):
        """
        Low-level function to demodulate signal using biexponential decay and remove artifact, operates on numpy
        arrays
        Parameters
        ----------
        timestamps: np.array with timestamps
        signal: np.array with signal
        isosbestic: np.array with isosbestic signal
        steps: if True, returns intermediate steps

        Returns
        -------

        """
        pass  # TODO

    def remove_artifact(self, artifact_signal=None):
        """
        Low-level function to remove artifact from signal, operates on numpy arrays
        :param artifact_signal: np.array with artifact signal, if None, raw signal is used
        :return: np.array with artifact removed signal
        """

        if artifact_signal is None:
            artifact_signal = self.data.signal.to_numpy()

        res = pd.DataFrame()

        res["isosbestic_airPLS"] = self.airPLS(
            self.data.isosbestic.to_numpy(),
            lambda_=self.lambda_artifact,
            porder=self.porder_artifact,
            itermax=self.itermax_artifact,
        )
        res["signal_airPLS"] = self.airPLS(
            artifact_signal,
            lambda_=self.lambda_artifact,
            porder=self.porder_artifact,
            itermax=self.itermax_artifact,
        )

        res["signal_wo_slope"] = artifact_signal - res.signal_airPLS
        res["isosbestic_wo_slope"] = self.data.isosbestic.to_numpy() - res.isosbestic_airPLS

        # Align reference signal to calcium signal using non-negative robust linear regression
        lin = Lasso(
            alpha=0.0001,
            precompute=True,
            max_iter=1000,
            positive=True,
            random_state=9999,
            selection="random",
        )

        # zero-center both signals
        res["isosbestic_centered"] = res.isosbestic_wo_slope - np.median(res.isosbestic_wo_slope)
        res["signal_centered"] = res.signal_wo_slope - np.median(res.signal_wo_slope)

        n = len(res.isosbestic_centered)
        lin.fit(res.isosbestic_centered.to_numpy().reshape(n, 1), res.signal_centered.to_numpy().reshape(n, 1))
        res["isosbestic_fit"] = lin.predict(res.isosbestic_centered.to_numpy().reshape(n, 1)).reshape(
            n,
        )

        res["signal_wo_artifacts"] = artifact_signal - res.isosbestic_fit

        if self.steps:
            return res
        return res.signal_wo_artifacts

    def demodulate_with_airPLS(self, data):
        """
        Low-level function to demodulate signal using airPLS; estimates a baseline and calculates dFF
        Parameters
        ----------
        data
            pd.DataFrame with raw signal and timestamps

        Returns
        -------
        pd.Series with demodulated signal or df with intermediate steps as additional columns

        """
        # Remove slope using airPLS algorithm
        data["baseline_fit"] = self.airPLS(data.signal, self.lambda_, self.porder, self.itermax)
        if any(data.baseline_fit <= 0):
            raise ValueError(
                "Negative or zero values in baseline fit, try to decrease lambda_, check if there is "
                "any fluorescence (if your measured values are very close to zero, there might not be)"
            )
        data["dFF"] = (data.signal - data.baseline_fit) / data.baseline_fit

        if self.steps:
            return data
        return data.dFF

    def demodulate_with_biexponential_decay(timestamps, signal, per, steps=False):
        """
        Low-level function to demodulate signal using biexponential decay
        :param df: df with raw data, containing columns 'Signal' and 'Reference'
        :param per: int, period of the signal
        :param steps: if True, returns intermediate steps
        :return: pd.Series with demodulated signal
        """

        # Demodulate signal
        df["Demodulated"] = df["Signal"] / df["Reference"]
        df["Demodulated"] = df["Demodulated"].rolling(per, center=True).mean()
        return df

    @staticmethod
    def smooth_signal(x, window_len=10, window="flat"):
        """
        smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
        :param x: np array, the input signal
        :param window_len: the dimension of the smoothing window; should be an odd integer
        :param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    'flat' window will produce a moving average smoothing.
        :return: array, the smoothed signal
        """

        x = np.array(x)

        if x.ndim != 1:
            raise (ValueError, "smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise (ValueError, "Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            raise (
                ValueError,
                "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'",
            )

        s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]

        if window == "flat":  # Moving average
            w = np.ones(window_len, "d")
        else:
            w = eval("np." + window + "(window_len)")

        y = np.convolve(w / w.sum(), s, mode="valid")

        y = y[(int(window_len / 2) - 1) : -int(window_len / 2)]
        if len(x) != len(y):
            y = y[: len(x)]
        return y

    """
    # TODO: delete
    def zdFF_airPLS(df, smooth_win=10, remove=200, lambd=5e4, porder=1, itermax=50):
        Low-level implementation of the airPLS Pipeline
        Handles values in df as a single datastream
        Calculates z-score dF/F signal based on fiber photometry calcium-independent
        and calcium-dependent signals
            :param df: df with raw data, containing columns 'Signal' and 'Reference'
            :param smooth_win: window for moving average smooth, integer
            :param remove: the beginning of the traces with a big slope one would like to remove, integer
            :param lambd: parameter for airPLS. The larger lambda is,
                    the smoother the resulting background
            :param porder: adaptive iteratively reweighted penalized least squares for baseline fitting
            :param itermax: maximum iteration times
            :return: df with 'zdFF (airPLS)' as an additional column
        #

        # remove beginning and end of recording
        df = df[(df.FrameCounter > remove) & (df.FrameCounter < max(df.FrameCounter) - remove)]
        # df = df.drop(np.arange(1, remove)).drop(np.arange(max(df.index) - remove, max(df.index)))

        # Smooth signal
        reference = smooth_signal(df["Reference"], smooth_win)
        signal = smooth_signal(df["Signal"], smooth_win)

        # Remove slope using airPLS algorithm
        reference -= airPLS(reference, lambda_=lambd, porder=porder, itermax=itermax)
        signal -= airPLS(signal, lambda_=lambd, porder=porder, itermax=itermax)

        # Standardize signals
        # TODO: why use median and not mean?
        reference = (reference - np.median(reference)) / np.std(reference)
        signal = (signal - np.median(signal)) / np.std(signal)

        # Align reference signal to calcium signal using non-negative robust linear regression
        lin = Lasso(
            alpha=0.0001,
            precompute=True,
            max_iter=1000,
            positive=True,
            random_state=9999,
            selection="random",
        )
        n = len(reference)
        lin.fit(reference.reshape(n, 1), signal.reshape(n, 1))
        reference = lin.predict(reference.reshape(n, 1)).reshape(
            n,
        )

        df["zdFF (airPLS)"] = signal - reference
        return df


    Ocober 2019 Ekaterina Martianova ekaterina.martianova.1@ulaval.ca

    Reference:
      (1) Martianova, E., Aronson, S., Proulx, C.D. Multi-Fiber Photometry
          to Record Neural Activity in Freely Moving Animal. J. Vis. Exp.
          (152), e60278, doi:10.3791/60278 (2019)
          https://www.jove.com/video/60278/multi-fiber-photometry-to-record-neural-activity-freely-moving

    airPLS.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
    Baseline correction using adaptive iteratively reweighted penalized least squares

    This program is a translation in python of the R source code of airPLS version 2.0
    by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls

    Reference:
    Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive iteratively
    reweighted penalized least squares. Analyst 135 (5), 1138-1146 (2010).

    LICENCE
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    """

    @staticmethod
    def whittaker_smooth(x, w, lambda_, differences=1):
        """
        Penalized least squares algorithm for background fitting
            :param x: input data (i.e. chromatogram of spectrum)
            :param w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
            :param lambda_: parameter that can be adjusted by user. The larger lambda is,
                     the smoother the resulting background
            :param differences: integer indicating the order of the difference of penalties
            :return: np array of whittaker smooth
        """

        x = np.array(x)
        w = np.array(w)

        X = np.matrix(x)
        m = X.size
        E = eye(m, format="csc")
        D = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
        W = diags(w, 0, shape=(m, m))
        A = csc_matrix(W + (lambda_ * D.T * D))
        B = csc_matrix(W * X.T)
        background = spsolve(A, B)
        return np.array(background)

    @staticmethod
    def airPLS(x, lambda_=1e4, porder=1, itermax=50):
        """
        Adaptive iteratively reweighted penalized least squares for baseline fitting
            :param x: input data (i.e. chromatogram of spectrum)
            :param lambda_: parameter that can be adjusted by user. The larger lambda is,
                     the smoother the resulting background, z
            :param porder: adaptive iteratively reweighted penalized least squares for baseline fitting
            :param itermax: maximal amount of iterations
            :return: the fitted background vector
        """
        # convert to numpy docstring

        x = np.array(x)

        m = x.shape[0]
        w = np.ones(m)
        if itermax < 1:
            raise ValueError("itermax must be a positive integer")
        for i in range(1, itermax + 1):
            z = _Demodulator.whittaker_smooth(x, w, lambda_, porder)
            d = x - z
            dssn = np.abs(d[d < 0].sum())
            if dssn < 0.001 * (abs(x)).sum() or i == itermax:
                if i == itermax:
                    print("WARING max iteration reached!")
                break
            w[
                d >= 0
            ] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
            w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
            w[0] = np.exp(i * (d[d < 0]).max() / dssn)
            w[-1] = w[0]
        return z


if __name__ == "__main__":
    df = pd.read_csv("example_data.csv")
    df.Signal -= 1 / 255
    df.Reference -= 1 / 255
    # ex = df[(df.injection=='baseline') & (df.mouse=='A1') & (df.Channel==560)].copy()
    ex = df
    ex.Signal = ex.Signal.rolling(8, center=True, min_periods=1).mean()
    ex.Reference = ex.Reference.rolling(8, center=True, min_periods=1).mean()

    # plt.plot(ex.index, ex.Signal)
    # plt.show()
    # get df column names
    ex = ex.set_index(["injection", "mouse"])
    # out = demodulate(ex, 'FrameCounter', 'Signal', None, ['injection', 'mouse'])

    out = demodulate(
        data=pd.DataFrame(
            {
                "timestamps": [1, 2, 3, 4, 5],
                "signal": [1, 2, 3, 4, 5],
                "session": ["a", "a", "a", "a", "a"],
            }
        ),
        timestamps="timestamps",
        signal="signal",
        by="session",
    )

    df["dFF"] = out

    print("halt")
