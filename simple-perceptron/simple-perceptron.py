import numpy as np
import matplotlib.pyplot as plt

# データの準備
rng = np.random.RandomState(123)    # 生成するデータを制限する

d = 2   # データの次元
N = 10  # 各パターンのデータ数
mean = 5    # ニューロンが発火する(fx=1 x>0 )データの平均値。データを簡単に分けてるために設定している

# それぞれのデータを生成する
x1 = rng.randn(N, d) + np.array([0, 0])
x2 = rng.randn(N, d) + np.array([mean, mean])

# 生成したデータをまとめて処理するため、x1,x2をまとめる
x = np.concatenate((x1, x2), axis=0)

# 生成されたデータを表示
plt.plot(x1[0], 'ro', label='x1')
plt.plot(x2, 'k^', label='x2')
plt.title('normal-distribution of x1,x2')
plt.legend()
plt.show()
