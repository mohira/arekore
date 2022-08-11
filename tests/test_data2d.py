import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from arekore.data2d import Data2d


class TestData2d:

    @pytest.fixture()
    def data(self) -> Data2d:
        x = np.arange(1, 100 + 1, 1)
        y = np.arange(101, 200 + 1, 1)

        return Data2d(x, y)

    class Test統計量計算:
        class TestXとYの最小最大が計算できる:
            def test_xy_minmax(self, data: Data2d):
                assert (data.xmin(), data.xmax(), data.ymin(), data.ymax()) == (1, 100, 101, 200)

        class Test相関係数を計算できる:
            def test_xy_minmax(self, data: Data2d):
                assert data.cor() == pytest.approx(1)

        class Test回帰係数と切片が計算される:
            def test_xy_minmax(self, data: Data2d):
                assert (data.intercept, data.coef) == (pytest.approx(100), pytest.approx(1))

    class Test表への変換:
        def test_DataFrameに変換できる(self):
            x = np.array([1, 2, 3])
            y = np.array([4, 5, 6])
            want = pd.DataFrame(data={'x': x, 'y': y})

            d = Data2d(x, y)
            got = d.to_df()

            assert_frame_equal(got, want)

        def test_katex対応のマークダウンに変換できる(self):
            want = r"""| x         | y           |
|:----------|:------------|
| \\(1.1\\) | \\(4.568\\) |
| \\(2.1\\) | \\(5.568\\) |
| \\(3.1\\) | \\(6.568\\) |"""

            x = np.array([1.1234, 2.1234, 3.1234])
            y = np.array([4.5678, 5.5678, 6.5678])
            d = Data2d(x, y)

            got = d.to_md(x_decimal=1, y_decimal=3, katex=True)

            assert got == want

    class TestError:
        @pytest.mark.parametrize('x, y', [
            (np.array([1, 2, 3]), np.array([1, 2, 3, 4])),
            (np.array([1, 2, 3]), np.array([[1, 2], [3]], dtype='object')),
        ])
        def test_xとyのデータ数が一致しなくてはならない(self, x: np.ndarray, y: np.ndarray):
            with pytest.raises(ValueError):
                Data2d(x, y)
