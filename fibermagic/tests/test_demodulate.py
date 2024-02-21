import pandas as pd
import pytest

from ..core.demodulate import _Demodulator


@pytest.fixture(
    params=[
        dict(
            data=pd.DataFrame(
                {
                    "timestamps": [1, 2, 3, 4, 5],
                    "signal": [1, 2, 3, 4, 5],
                }
            ),
            timestamps="timestamps",
            signal="signal",
        ),
        dict(
            data=pd.DataFrame(
                {
                    "foo": [1, 2, 3, 4, 5],
                    "faa": [1, 2, 3, 4, 5],
                }
            ),
            timestamps="foo",
            signal="faa",
        ),
        # test single by group
        dict(
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
        ),
        # test multiple by groups
        dict(
            data=pd.DataFrame(
                {
                    "timestamps": [1, 2, 3, 4, 5],
                    "signal": [1, 2, 3, 4, 5],
                    "session": ["a", "a", "a", "b", "b"],
                }
            ),
            timestamps="timestamps",
            signal="signal",
            by="session",
        ),
        # test multiple by columns
        dict(
            data=pd.DataFrame(
                {
                    "timestamps": [1, 2, 3, 4, 5],
                    "signal": [1, 2, 3, 4, 5],
                    "session": ["a", "a", "a", "b", "b"],
                    "mouse": ["m1", "m3", "m3", "m2", "m2"],
                }
            ),
            timestamps="timestamps",
            signal="signal",
            by=["session", "mouse"],
        ),
        # test by in index
        dict(
            data=pd.DataFrame(
                {
                    "timestamps": [1, 2, 3, 4, 5],
                    "signal": [1, 2, 3, 4, 5],
                    "session": ["a", "a", "a", "b", "b"],
                    "mouse": ["m1", "m3", "m3", "m2", "m2"],
                }
            ).set_index(["session", "mouse"]),
            timestamps="timestamps",
            signal="signal",
            by=["session", "mouse"],
        ),
        # test only artifact removal
        dict(
            data=pd.DataFrame(
                {
                    "timestamps": [1, 2, 3, 3, 3, 4, 5],
                    "signal": [1, 2, 3, 3, 3, 4, 5],
                    "isosbestic": [1, 2, 4, 3, 3, 4, 5],
                    "session": ["a", "a", "a", "a", "a", "a", "a"],
                    "mouse": ["m1", "m1", "m1", "m1", "m1", "m1", "m1"],
                }
            ).set_index(["session", "mouse"]),
            timestamps="timestamps",
            signal="signal",
            isosbestic="isosbestic",
            by=["session", "mouse"],
            method=None,
        ),
        # test demodulation plus artifact removal
        dict(
            data=pd.DataFrame(
                {
                    "timestamps": [1, 2, 3, 3, 3, 4, 5],
                    "signal": [1, 2, 3, 3, 3, 4, 5],
                    "isosbestic": [1, 2, 4, 3, 3, 4, 5],
                    "session": ["a", "a", "a", "a", "a", "a", "a"],
                    "mouse": ["m1", "m1", "m1", "m1", "m1", "m1", "m1"],
                }
            ).set_index(["session", "mouse"]),
            timestamps="timestamps",
            signal="signal",
            isosbestic="isosbestic",
            by=["session", "mouse"],
        ),
        # test biexponential decay
        dict(
            data=pd.DataFrame(
                {
                    "timestamps": [1, 2, 3, 3, 3, 4, 5],
                    "signal": [6, 5, 4, 3, 2, 1, 0],
                    "session": ["a", "a", "a", "a", "a", "a", "a"],
                    "mouse": ["m1", "m1", "m1", "m1", "m1", "m1", "m1"],
                }
            ).set_index(["session", "mouse"]),
            timestamps="timestamps",
            signal="signal",
            isosbestic=None,
            by=["session", "mouse"],
            method="biexponential decay",
        ),
        # test biexponential decay with smoothing
        dict(
            data=pd.DataFrame(
                {
                    "timestamps": [1, 2, 3, 3, 3, 4, 5, 6],
                    "signal": [
                        6,
                        5,
                        4,
                        3,
                        2,
                        1,
                        1,
                        1,
                    ],
                    "isosbestic": [
                        6,
                        5,
                        4,
                        3,
                        2,
                        1,
                        0,
                        0,
                    ],
                    "session": ["a", "a", "a", "a", "a", "a", "a", "a"],
                    "mouse": ["m1", "m1", "m1", "m1", "m1", "m1", "m1", "m1"],
                }
            ).set_index(["session", "mouse"]),
            timestamps="timestamps",
            signal="signal",
            isosbestic="isosbestic",
            by=["session", "mouse"],
            method="biexponential decay",
            smooth=2,
        ),
        # test airPLS with smoothing
        dict(
            data=pd.DataFrame(
                {
                    "timestamps": [1, 2, 3, 3, 3, 4, 5, 6],
                    "signal": [
                        6,
                        5,
                        4,
                        3,
                        2,
                        1,
                        1,
                        1,
                    ],
                    "isosbestic": [
                        6,
                        5,
                        4,
                        3,
                        2,
                        1,
                        0,
                        0,
                    ],
                    "session": ["a", "a", "a", "a", "a", "a", "a", "a"],
                    "mouse": ["m1", "m1", "m1", "m1", "m1", "m1", "m1", "m1"],
                }
            ).set_index(["session", "mouse"]),
            timestamps="timestamps",
            signal="signal",
            isosbestic="isosbestic",
            by=["session", "mouse"],
            method="airPLS",
            smooth=2,
        ),
        # test dicts as data input
        dict(
            data={
                "timestamps": [1, 2, 3, 4, 5],
                "signal": [1, 2, 3, 4, 5],
            },
            timestamps="timestamps",
            signal="signal",
        ),
        # test z-score
        dict(
            data=pd.DataFrame(
                {
                    "timestamps": [1, 2, 3, 3, 3, 4, 5, 6],
                    "signal": [
                        6,
                        5,
                        4,
                        3,
                        2,
                        1,
                        1,
                        1,
                    ],
                    "isosbestic": [
                        6,
                        5,
                        4,
                        3,
                        2,
                        1,
                        0,
                        0,
                    ],
                    "session": ["a", "a", "a", "a", "a", "a", "a", "a"],
                    "mouse": ["m1", "m1", "m1", "m1", "m1", "m1", "m1", "m1"],
                }
            ).set_index(["session", "mouse"]),
            timestamps="timestamps",
            signal="signal",
            isosbestic="isosbestic",
            by=["session", "mouse"],
            method="airPLS",
            smooth=2,
            standardize=True,
        ),
    ]
)
def get_demodulator(request):
    return _Demodulator(**request.param), request.param["data"]


@pytest.fixture
def grouped_data():
    return pd.DataFrame(
        {
            "timestamps": [1, 2, 3, 4, 5],
            "signal": [1, 2, 3, 4, 5],
        }
    )


def test_demodulate_with_airPLS_returns_series(get_demodulator, grouped_data):
    demodulator, _ = get_demodulator
    demodulated = demodulator.demodulate_with_airPLS(grouped_data)
    assert isinstance(demodulated, pd.Series)


def test_returns_same_index(get_demodulator):
    demodulator, data = get_demodulator
    demodulated = demodulator()
    if isinstance(data, pd.DataFrame):
        assert demodulated.index.equals(data.index)
