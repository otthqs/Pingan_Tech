import numpy as np
import pandas as pd
from scipy.stats import rankdata

pd.set_option("use_inf_as_na", True)


def con(x, y, z):
    """

    :param x:
    :param y:
    :param z:
    :return: 若 X 为 true 则返回 Y，否则返回 Z(同 C 程序中定义)
    """
    data = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    if isinstance(y, (int, float)):
        data[x] = y
    else:
        data[x] = y[x]
    if isinstance(z, (int, float)):
        data[~x] = z
    else:
        data[~x] = z[~x]
    return data


def minp(d):
    """

    :param d: Pandas rolling 的 window
    :return: 返回值为 int，对应 window 的 min_periods
    """
    if not isinstance(d, int):
        d = int(d)
    if d <= 10:
        return d - 1
    else:
        return d * 2 // 3


def sign(x):
    """

    :param x:
    :return:
    """
    return np.sign(x)


def rank(x):
    """

    :param x: 代表 N 只个股在某指定截面日的因子值
    :return: 返回值为向量，其中第 i 个元素为𝑋𝑖在向量 X 中的分位数
    """
    return x.rank(axis = 1).sub(0.5).div(x.count(axis = 1))


def delay(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: d 天以前的 X 值
    """
    return x.shift(d)


def correlation(x, y, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param y:
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天 𝑋_i 值构成的时序数列和 𝑌_i 值构成的时序数列的相关系数
    """
    return x.rolling(window=int(d), min_periods=minp(d)).corr(y)


def covariance(x, y, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param y:
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天𝑋 值构成的时序数列和𝑌 值构成的时序数列的协方差
    """
    return x.rolling(window=int(d), min_periods=minp(d)).cov(y)


def scale(x, a=1):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param a:
    :return: 返回值为向量 a*X/sum(abs(x))，a 的缺省值为 1，一般 a 应为正数
    """
    return x.mul(a).div(np.abs(x).sum(axis=1), axis=0)


def delta(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量 X - delay(X, d)
    """
    return x.diff(int(d))


def signedpower(x, a):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param a:
    :return: 返回值为向量 sign(X).*(abs(X).^a)，其中.*和.^两个运算符代表向量中对应元素相乘、元素乘方
    """
    return np.sign(x) * (np.power(np.abs(x), a))


def decay_linear(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天 𝑋_𝑖 值构成的时序数列的加权平均值，权数为 d, d – 1, ..., 1
    (权数之和应为 1，需进行归一化处理)，其中离现在越近的日子权数越大
    """

    weight = np.arange(0, int(d)) + 1

    res = (x.rolling(window = int(d), min_periods = minp(d)).apply(lambda z: np.nansum(z * weight[-len(z):]) / weight[-len(z):][~np.isnan(z)].sum(), raw = True))

    return res



def ts_min(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天 𝑋_𝑖 值构成的时序数列中最小值
    """
    return x.rolling(window=int(d), min_periods=minp(d), axis=0).min()


def ts_max(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天 𝑋_𝑖 值构成的时序数列中最大值
    """
    return x.rolling(window=int(d), min_periods=minp(d), axis=0).max()


def ts_argmin(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天 𝑋_𝑖 值构成的时序数列中最小值出现的位置
    """
    return x.rolling(window=int(d), min_periods=minp(d), axis=0).apply(lambda z: np.nan if np.all(np.isnan(z)) else np.nanargmin(z), raw = True) + 1


def ts_argmax(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天 𝑋_𝑖 值构成的时序数列中最大值出现的位置
    """
    return x.rolling(window=int(d), min_periods=minp(d), axis=0).apply(lambda z: np.nan if np.all(np.isnan(z)) else np.nanargmax(z), raw = True) + 1


def ts_rank(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天 𝑋_𝑖 值构成的时序数列中本截面日 𝑋_𝑖 值所处分位数
    """
    return x.rolling(int(d), min_periods = minp(d)).apply(
    lambda z: np.nan if np.all(np.isnan(z)) else ((rankdata(z[~np.isnan(z)])[-1] -1) * (len(z)-1) / (len(z[~np.isnan(z)]) - 1) + 1), raw = True)


def min_(x, y):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param y:
    :return: 若函数形式为 min(X, Y)，则返回值为向量，其中第 i 个元素为min(𝑋_i, 𝑌_i);若函数形式为 min(X, d)，
    则定义同 ts_min(X, d)。max 与 min 同理。
    """
    if type(y) is int:
        return ts_min(x, y)
    else:
        return np.minimum(x, y)


def max_(x, y):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param y:
    :return: 若函数形式为 min(X, Y)，则返回值为向量，其中第 i 个元素为min(𝑋_i, 𝑌_i);若函数形式为 min(X, d)，
    则定义同 ts_min(X, d)。max 与 min 同理。
    """
    if type(y) is int:
        return ts_max(x, y)
    else:
        return np.maximum(x, y)


def ts_sum(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列之和
    """
    return x.rolling(window=int(d), min_periods=minp(d)).sum()


def sum_(x, d=None):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 若函数形式为 sum(X, d)，则返回值为向量，其中第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列之和;
    若函数形式为 sum(X)，则返回值为一个数，为向量 X 中所有元素之和
    """
    if d is None:
        return x.sum()
    else:
        return ts_sum(x, d)


def ts_product(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列的连乘乘积
    """


    return np.log(np.exp(x).rolling(window = int(d), min_periods=minp(d), axis = 0).mean() * int(d))


def stddev(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列的标准差
    """
    return x.rolling(window=int(d), min_periods=minp(d), axis=0).apply(np.nanstd, raw=True)


def ts_stddev(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 返回值为向量，其中第 i 个元素为过去 d 天𝑋𝑖值构成的时序数列的标准差
    """
    return x.rolling(window = int(d), min_periods=minp(d), axis = 0).std()


def log(x):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :return:
    """
    # 注意: 不要改变原数据
    _x = x.copy(True)
    # RuntimeWarning: invalid value encountered in log
    _x[_x <= 0] = np.nan
    return np.log(_x)


def get_adv(x, d):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param d:
    :return: 个股过去 N 个交易日的平均成交量，例如 ADV20 代表过去 20 个交易日平均成交量
    """
    return x.rolling(window=int(d), min_periods=minp(d)).mean()


def div(x, y):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param y:
    :return: 返回值为向量，其中第 i 个元素为 𝑋_i * 𝑌_i (对应 matlab 中的点除)
    """
    return x.div(y)


def mul(x, y):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param y:
    :return: 返回值为向量，其中第 i 个元素为 𝑋_i / 𝑌_i (对应 matlab 中的点乘)
    """
    return x.mul(y)


def add(x, y):
    """

    :param x: X 的每一行代表 N 只个股在某指定截面日的因子值
    :param y:
    :return: 返回值为向量，其中第 i 个元素为 𝑋_i + 𝑌_i
    """
    return x + y
