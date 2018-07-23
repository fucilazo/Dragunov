mca = {"a": 1, "b": 2, "c": 3, "d": 4}
dicts = {v: k for k, v in mca.items()}
print(dicts)
# ======================================================
# 大小写合并
mcase = {'a': 10, 'b': 34, 'A': 7, 'Z': 3}
mcase_frequency = {
    k.lower(): mcase.get(k.lower(), 0) + mcase.get(k.upper(), 0)
    for k in mcase.keys()
    if k.lower() in ['a', 'b']
}
print(mcase_frequency)
# ======================================================
# 快速更换key和value
mcase = {'a': 10, 'b': 34}
mcase_frequency = {v: k for k, v in mcase.items()}
print(mcase_frequency)
