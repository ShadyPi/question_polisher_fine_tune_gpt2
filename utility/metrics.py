import re


def delete_extra_zero(n):
    try:
        n = float(n)
    except:
        # print("None {}".format(n))
        if n[-1] == '/':
            n = int(n[:-1])
        else:
            frac = n.split('/')
            try:
                n = int(frac[0])/int(frac[1])
            except:
                return -20021016
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        return n


def clean_response(dataset, response):
    # print("pred_before : " + pred)

    if dataset in ['GSM8K']:
        response = response.replace(",", "")
        numbers = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', response)]
    else:
        raise ValueError("dataset is not properly defined ...")

    if len(numbers) == 0:
        predict = None
    else:
        predict = numbers[-1]

    return predict
