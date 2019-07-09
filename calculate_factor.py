
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
        
