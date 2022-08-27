import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy import stats

from arekore.dummy import multi_values_specified_correlation, xy_specified_cor


class TestDummy:
    @pytest.mark.parametrize('r, n', (
        (0.8, 100),
        (0.0, 100),
        (-0.9, 100),
    ))
    def test_任意の相関係数をもつ2変量データを作成できる(self, r: float, n: int):
        x, y = xy_specified_cor(r, n)

        assert stats.pearsonr(x, y).statistic == pytest.approx(r)

    class Test多変量データの生成:
        class Test_任意の相関行列をもつ多変量データからサンプリングできる:
            @pytest.mark.parametrize('R', (
                np.array([[1.0, 0.3, 0.4],
                          [0.3, 1.0, 0.5],
                          [0.4, 0.5, 1.0]]),
                np.array([[1.0, 0.3, -0.4],
                          [0.3, 1.0, -0.5],
                          [-0.4, -0.5, 1.0]])
            ))
            def test_sampling(self, R: np.ndarray):
                X = multi_values_specified_correlation(R, n=10 ** 5)

                actual_correlation = pd.DataFrame(X).corr()

                assert_almost_equal(R, actual_correlation, decimal=2)

        class TestError:
            @pytest.mark.parametrize('R', (
                np.array([[2, 0.3],
                          [0.4, 2]]),
                np.array([[-2, 0.3],
                          [0.4, -2]]),
            ))
            def test_相関係数の定義域を満たしていなければならない(self, R: np.ndarray):
                with pytest.raises(ValueError):
                    multi_values_specified_correlation(R, n=3)

            def test_相関行列が対称行列になっていなければならない(self):
                R = np.array([
                    [1.0, 0.33],
                    [0.44, 1.0],
                ])
                with pytest.raises(ValueError):
                    multi_values_specified_correlation(R, n=1)
