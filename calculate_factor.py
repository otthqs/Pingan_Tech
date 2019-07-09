
# 基于已有数据进行辅助指标的计算:
# 1.短期和长期指数平滑移动平均线已提前计算出结果并存为emas（12日）以及emal（26日）



def calculate_factor(factor):
    """
    factor: str, the name of factor
    return: pd.DataFrame, values are dummy valriables of 1, 0 and -1
    Note:
    """

    factor = factor.upper() # Ignore the upper and lower case

    if factor == "MACD":
        emas = cls.ewm(min_periods = 1, span = 12, ignore_na = True , adjust = False).mean()
        emal = cls.ewm(min_periods = 1, span = 26, ignore_na = True , adjust = False).mean()
        dif = emas - emal
        dea = dif.ewm(min_periods = 1, span = 9, ignore_na = True, adjust = False).mean()
        macd = (dif - dea) * 2
        result = ((dif > 0) & (macd > 0) & (dif.shift(1) < macd) & (dif > macd)) * 1 + ((dif > 0) & (macd > 0) & (dif.shift(1) > macd) & (dif < macd)) * -1
