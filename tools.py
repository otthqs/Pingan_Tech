"""
Some auxiliary functions of factors
"""

def check_signals(data):
    """
    data -> DataFrame: single factor results

    return -> dictionary: count number of single signals, two signals in a row and three signals in a row
    """

    one = ((res == 1) & (res.shift(-1) !=1 )).sum().sum()
    two = ((res == 1) & (res.shift(-1) == 1) & (res.shift(-2) != 1)).sum().sum()
    three = ((res == 1) & (res.shift(-1) == 1) & (res.shift(-2) == 1) & (res.shift(-3) != 1)).sum().sum()

    total = data.sum().sum()

    dic = {}
    dic["single"] = (one, one / total)
    dic["two_in_a_row"] = (two, two / total)
    dic["three_in_a_row"] = (three, three / total)

    return dic
