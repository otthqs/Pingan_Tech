def quant_factor(data,num):
    """
    Assign continuously factor value to each group before backtest
    data: np.array
    num : int
    return : dictionary - key is the number of group, value is array

    """
    total = len(data) - np.isnan(data).sum() # effective factors in all stocks

    weight = total/num # Calculate the expected weight in each group

    col = [i for i in range(len(data))] # Create the index to track the stock after sort method

    result = {key:np.array([0.0]*len(data)) for key in range(1,num+1)} # initia the return dictionary, which is weight corresponding to each stock

    data_ = list(zip(data,col)) # Build a map between stock and the original index
    data_.sort()

#  To assign sorted data into groups, each group has the same value.
    group = []

    sub = []
    sub.append(data_[0][1])
    before = data_[0][0]

    for i in range(1,len(data_)):
        if not np.isnan(data_[i][0]):
            if data_[i][0] == before:
                sub.append(data_[i][1])

            else:
                group.append(sub)
                before = data_[i][0]
                sub = []
                sub.append(data_[i][1])

    if sub:
        group.append(sub)

    short = {key: weight for key in range(1,num+1)} # The weight that is still need or the weigh left.

    ind = 1

    for sub in group: # each sub is a pool of stock with different weight.
        left = len(sub)

        while left and ind <= num:
            if short[ind] >= left:
                for ind_ in sub:
                    result[ind][ind_] += left/len(sub)
                short[ind] -= left
                left = 0

            elif short[ind] < left:
                for ind_ in sub:
                    result[ind][ind_] += short[ind]/len(sub)
                left -= short[ind]
                short[ind] = 0
                ind += 1

    return result
