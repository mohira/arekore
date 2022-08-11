import numpy as np
import pytest
from numpy.testing import assert_array_equal

from arekore.data1d import Data1d


class TestData1d:
    @pytest.fixture()
    def dummy_bins(self) -> np.ndarray:
        return np.array([])

    class Test統計量算:
        class Test最小値を計算できる:
            @pytest.mark.parametrize('rawdata, want', [
                pytest.param(np.array([1, 2, 3]), 1),
            ])
            def test_min(self, rawdata: np.ndarray, want: float, dummy_bins: np.ndarray):
                assert Data1d(rawdata, dummy_bins).min() == want

        class Test最大値を計算できる:
            @pytest.mark.parametrize('rawdata, want', [
                pytest.param(np.array([1, 2, 3]), 3),
            ])
            def test_mean(self, rawdata: np.ndarray, want: float, dummy_bins: np.ndarray):
                assert Data1d(rawdata, dummy_bins).max() == want

        class Test中央値を計算できる:
            @pytest.mark.parametrize('rawdata, want', [
                (np.array([1, 2, 3, 4, 5]), 3),
                (np.array([1, 2, 2, 3]), 2),
                (np.array([1, 2, 3, 4]), 2.5),
            ])
            def test_median(self, rawdata: np.ndarray, want: float, dummy_bins: np.ndarray):
                assert Data1d(rawdata, dummy_bins).median() == pytest.approx(want)

        class Test平均値を計算できる:
            @pytest.mark.parametrize('rawdata, want', [
                (np.array([1, 2, 1]), 1.333333),
            ])
            def test_mean(self, rawdata: np.ndarray, want: float, dummy_bins: np.ndarray):
                assert Data1d(rawdata, dummy_bins).mean() == pytest.approx(want)

        class Test標準偏差を計算できる:
            @pytest.mark.parametrize('rawdata, want', [
                (np.array([1, 2, 3]), 0.816496580927726),
            ])
            def test_std(self, rawdata: np.ndarray, want: float, dummy_bins: np.ndarray):
                assert Data1d(rawdata, dummy_bins).std() == pytest.approx(want)

        class Test最頻値:
            class Test離散量における最頻値計算に対応している:
                @pytest.mark.parametrize('rawdata, want', [
                    pytest.param(np.array([1, 2, 2, 3]), np.array([2]), id='single mode'),
                    pytest.param(np.array([1, 2, 3]), np.array([1, 2, 3]), id='multi mode'),
                ])
                def test_mode_for_not_bins(self, rawdata: np.ndarray, want: int, dummy_bins: np.ndarray):
                    d = Data1d(rawdata, dummy_bins)

                    assert_array_equal(d.mode(), want)

            class Test連続量における最頻値計算に対応している:
                def test_mode_with_bins_single(self):
                    rawdata = np.array([40.1,
                                        50.1, 50.2,
                                        60.1, 60.2,
                                        70.1, 70.2, 70.3, 70.4,
                                        80.1, 80.2,
                                        90.1, 90.2])
                    want = np.array([75])

                    target_bins = np.array([40, 50, 60, 70, 80, 90, 100])
                    mode = Data1d(rawdata, bins=target_bins).mode_with_bins()

                    assert_array_equal(mode, want)

                def test_mode_with_bins_multi(self):
                    rawdata = np.array([40.1,
                                        50.1, 50.2, 50.3,
                                        60.1, 60.2,
                                        70.1, 70.2, 70.3,
                                        80.1, 80.2, 80.3,
                                        90.1, 90.2])
                    want = np.array([55, 75, 85])

                    bins = np.array([40, 50, 60, 70, 80, 90, 100])
                    mode = Data1d(rawdata, bins=bins).mode_with_bins()

                    assert_array_equal(mode, want)

                def test_mode_with_bins_multi_自分で指定もできる(self, dummy_bins: np.ndarray):
                    rawdata = np.array([40.1, 50.1, 50.2, 60.1, 60.2, 60.3,
                                        70.1, 80.1, 80.2, 90.1, 90.2, 90.3])
                    want = np.array([55, 85])

                    target_bins = np.array([40, 70, 100])
                    mode = Data1d(rawdata, bins=dummy_bins).mode_with_bins(target_bins)

                    assert_array_equal(mode, want)

        class Test四分位数:
            @pytest.fixture()
            def d(self, dummy_bins: np.ndarray) -> Data1d:
                rawdata = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

                return Data1d(rawdata, dummy_bins)

            def test_四分位数と四分位範囲を計算できる(self, d: Data1d):
                assert (d.q1(), d.q3(), d.iqr()) == (3.25, 7.75, 4.5)

            def test_ヒンジ計算ができる(self, d: Data1d):
                assert (d.q1(hinge=True), d.q3(hinge=True), d.iqr(hinge=True)) == (3, 8, 5)

    class Test度数分布表:
        @pytest.fixture()
        def rawdata(self) -> np.ndarray:
            return np.array([40.1,
                             50.1, 50.2,
                             60.1, 60.2, 60.3,
                             70.1, 70.2, 70.3, 70.4])

        def test_度数分布表をつくれる(self, rawdata: np.ndarray):
            want = """               階級   階級値  度数  累積度数  相対度数  累積相対度数
0  (39.999, 50.0]  45.0   1     1   0.1     0.1
1    (50.0, 60.0]  55.0   2     3   0.2     0.3
2    (60.0, 70.0]  65.0   3     6   0.3     0.6
3    (70.0, 80.0]  75.0   4    10   0.4     1.0
4    (80.0, 90.0]  85.0   0    10   0.0     1.0
5   (90.0, 100.0]  95.0   0    10   0.0     1.0"""

            d = Data1d(rawdata, bins=np.array([40, 50, 60, 70, 80, 90, 100]))
            assert str(d.freq_table()) == want

        def test_度数分布表のマークダウンをつくれる(self, rawdata: np.ndarray):
            want = """| 階級             |   階級値 |   度数 |   累積度数 |   相対度数 |   累積相対度数 |
|:---------------|------:|-----:|-------:|-------:|---------:|
| (39.999, 50.0] |    45 |    1 |      1 |    0.1 |      0.1 |
| (50.0, 60.0]   |    55 |    2 |      3 |    0.2 |      0.3 |
| (60.0, 70.0]   |    65 |    3 |      6 |    0.3 |      0.6 |
| (70.0, 80.0]   |    75 |    4 |     10 |    0.4 |      1   |
| (80.0, 90.0]   |    85 |    0 |     10 |    0   |      1   |
| (90.0, 100.0]  |    95 |    0 |     10 |    0   |      1   |"""

            d = Data1d(rawdata, bins=np.array([40, 50, 60, 70, 80, 90, 100]))
            assert d.freq_table_as_md() == want

        def test_度数分布表のKatex対応マークダウンをつくれる(self, rawdata: np.ndarray):
            want = r"""| 階級                   | 階級値         | 度数      | 累積度数     | 相対度数       | 累積相対度数     |
|:---------------------|:------------|:--------|:---------|:-----------|:-----------|
| \\((39.999, 50.0]\\) | \\(45.00\\) | \\(1\\) | \\(1\\)  | \\(0.10\\) | \\(0.10\\) |
| \\((50.0, 60.0]\\)   | \\(55.00\\) | \\(2\\) | \\(3\\)  | \\(0.20\\) | \\(0.30\\) |
| \\((60.0, 70.0]\\)   | \\(65.00\\) | \\(3\\) | \\(6\\)  | \\(0.30\\) | \\(0.60\\) |
| \\((70.0, 80.0]\\)   | \\(75.00\\) | \\(4\\) | \\(10\\) | \\(0.40\\) | \\(1.00\\) |
| \\((80.0, 90.0]\\)   | \\(85.00\\) | \\(0\\) | \\(10\\) | \\(0.00\\) | \\(1.00\\) |
| \\((90.0, 100.0]\\)  | \\(95.00\\) | \\(0\\) | \\(10\\) | \\(0.00\\) | \\(1.00\\) |"""

            d = Data1d(rawdata, bins=np.array([40, 50, 60, 70, 80, 90, 100]))
            assert d.freq_table_as_md(katex_mode=True) == want

    class TestError:
        @pytest.mark.parametrize('rawdata', [
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 2, 3]]),
            np.array([[1]]),
            np.array([]),
            # np.array([[1], []]) # 2次元だが非対応。どのみち計算時に例外となるので後回し
        ])
        def test_1次元データでない場合は不正とする(self, rawdata: np.ndarray, dummy_bins: np.ndarray):
            with pytest.raises(ValueError):
                Data1d(rawdata, bins=dummy_bins)
