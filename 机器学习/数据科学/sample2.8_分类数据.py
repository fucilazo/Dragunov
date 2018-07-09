import pandas as pd

# 分类等级映射到数值列表
categorical_feature = pd.Series(['sunny', 'cloudy', 'snowy', 'rainy', 'foggy'])
mapping = pd.get_dummies(categorical_feature)

print(mapping)
print(mapping['sunny'])
print(mapping['cloudy'])