import pytest
from scipy import stats

from arekore.dummy import xy_specified_cor


class TestDummy:
    @pytest.mark.parametrize('r, n', (
        (0.8, 100),
        (0.0, 100),
        (-0.9, 100),
    ))
    def test_任意の相関係数をもつ2変量データを作成できる(self, r: float, n: int):
        x, y = xy_specified_cor(r, n)

        assert stats.pearsonr(x, y).statistic == pytest.approx(r)
