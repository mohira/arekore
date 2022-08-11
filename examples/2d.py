import matplotlib.pyplot as plt

from arekore import dummy, viz
from arekore.data2d import Data2d, plot_regression_line, plot_scatter


def main():
    # ダミーデータ生成
    x, y = dummy.xy_specified_cor(r=0.8, n=50)

    # 2次元データオブジェクトの作成
    d = Data2d(x=x, y=y)

    # 散布図と回帰直線を描画
    fig, ax = viz.fig_ax(figsize=(8, 6))
    plot_scatter(ax, d)
    plot_regression_line(ax, d)

    plt.show()


if __name__ == '__main__':
    main()
