def Linear_interpolation(valueL, valueR, total_num):
    if valueL is None:
        return [valueR]
    else:
        value = []
        value_len = valueR - valueL
        for i in range(total_num):
            value.append(valueL + float(i + 1) / total_num * value_len)

        return value
