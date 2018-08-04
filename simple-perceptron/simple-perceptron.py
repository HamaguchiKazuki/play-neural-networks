import numpy as np
import matplotlib.pyplot as plt

# データの準備
rng = np.random.RandomState(123)    # 生成するデータを制限する

d = 2   # データの次元
N = 10  # 各パターンのデータ数
mean = 10  # ニューロンが発火する(fx=1 x>0 )データの平均値。データを簡単に分けてるために設定している

# それぞれのデータを生成する
x1 = rng.randn(N, d) + np.array([0, 0])
x2 = rng.randn(N, d) + np.array([mean, mean])

# 生成したデータをまとめて処理するため、x1,x2をまとめる
x = np.concatenate((x1, x2), axis=0)

# 重みベクトルとバイアスの初期化
w = np.zeros(d)
b = 0

# y = f(wx+b)
def y(x):
    return step(np.dot(w, x) + b)

# 発火したかどうかの確認
def step(x):
    return 1 * (x > 0)

# 教師データの定義
def t(i):
    if i < N:
        return 0
    else:
        return 1

# 誤り訂正学習法
while True:
    # パラメータ更新の処理
    classified = True
    for i in range(N * 2):
        # w = (t-y)x, b = (t-y)
        delta_w = ( t(i) - y(x[i]) ) * x[i]
        delta_b = ( t(i) - y(x[i]) )
        # w^k+1 = w^k + delta_w, b^k+1 = w^k + delta_b
        w = w + delta_w
        b = b + delta_b
        # delta_wは配列のためすべてが0か確認する必要がある。all(), Ture,Falseの掛け算
        classified *= all(delta_w==0) * (delta_b==0)
    if classified:
        break

# 結果とテスト
print('w:', w)
print('b:', b)
print('\nTest')
print(y([0,0]))
print(y([5,5]))


# 直線の式を作る
x_line = np.linspace(-6,15)
y1 = - ( w[0]*x_line/w[1] + b/w[1] )
print(y1)

# 生成されたデータを表示
plt.xlim([-4,15])
plt.ylim([-6,30])
plt.plot(x1, 'ro')
plt.plot(x2, 'bo')
plt.plot(x_line, y1, "r-")
plt.title('normal-distribution of x1,x2')
plt.legend()
plt.show()
