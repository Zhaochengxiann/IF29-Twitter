
### 用IQR来对数据进行可视化，获取相关合理的用于打标签的参数

import pandas as pd
import numpy as np

# 载入数据
df = pd.read_csv("../dataset/cleaned_data_all.csv")

# 将缺失值填充为0
df["follower_friend_ratio"] = df["follower_friend_ratio"].fillna(0)

features = [
    'followers_count','friends_count','statuses_count', 'mention_count', 'url_count', 'text_length',
    'follower_friend_ratio', 'active_hours', 'tweets_per_day','aggressiveness', 'visibility'
]

# 保证 features 中字段都在 df_clean 里
X = df[features]

# 检查各列的唯一值数量（如果有其他几乎恒定的列也可以考虑删）
print(X.nunique())


X = df[features]

#计算双侧IQR的方法
def calculate_iqr_bounds(column_data):
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

#计算单侧IQR的方法
def calculate_high_iqr_bound(column_data):
    Q3 = column_data.quantile(0.75)
    Q1 = column_data.quantile(0.25)
    IQR = Q3 - Q1
    return Q3 + 1.5 * IQR

# 使用百分位法计算上下限
def calculate_percentile_bounds(column_data, lower_percentile=1, upper_percentile=99):
    return column_data.quantile(lower_percentile/100), column_data.quantile(upper_percentile/100)

# 保存每个特征对应的异常检测方法与阈值
bounds = {}

# -------- 针对每列特征进行自动判断并计算对应异常值界限 --------
for col in features:
    uniq = X[col].nunique()       # 唯一值个数
    total = len(X[col])           # 总样本数
    
    # 情况1：如果是比值类特征或明显右偏的指标 → 用 IQR 单侧（只检测高值异常）
    if 'ratio' in col or col in ['tweets_per_day', 'aggressiveness']:
        method = 'IQR-High'
        upper = calculate_high_iqr_bound(X[col])
        bounds[col] = {'method': method, 'lower': None, 'upper': upper}
    
    # 情况2：如果唯一值较少且样本量很大 → 使用百分位方法控制异常比例
    elif uniq < 1000 and total > 100000:
        method = 'Percentile'
        lb, ub = calculate_percentile_bounds(X[col], 1, 99)  # 取1%-99%的范围
        bounds[col] = {'method': method, 'lower': lb, 'upper': ub}
    
    # 情况3：默认使用 IQR 双侧法检测异常（适用于多数连续变量）
    else:
        method = 'IQR-Both'
        lb, ub = calculate_iqr_bounds(X[col])
        bounds[col] = {'method': method, 'lower': lb, 'upper': ub}

# -------- 输出所有特征的异常值检测方法及阈值 --------

# 将结果转换为 DataFrame
bounds_df = pd.DataFrame.from_dict(bounds, orient='index')

# 重设索引名称为 feature（原本是列名）
bounds_df.index.name = 'feature'

# 显示结果
print(bounds_df)


'''

import pandas as pd
import numpy as np

# 载入数据
df = pd.read_csv("../dataset/cleaned_data_all.csv")

numeric_df = df.drop(columns=['user_id'])

#默认受数据处理那边代码的影响
#follower_friend_ratio为空
#tweets_per_day, aggressiveness, visibility 为 0

# 将0值设置为NaN，以便在IQR计算中被忽略
# columns_to_replace = ['tweets_per_day', 'aggressiveness', 'visibility']
# numeric_df[columns_to_replace] = numeric_df[columns_to_replace].replace(0, np.nan)

# 将空值设为0
#numeric_df['follower_friend_ratio'] = numeric_df['follower_friend_ratio'].fillna(0)

iqr_bounds = {}

for column in numeric_df.columns:
    # 这部分会识别是否为0记录总数，会对数据造成影响
    Q1 = numeric_df[column].quantile(0.25)
    Q3 = numeric_df[column].quantile(0.75)

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_bounds[column] = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

# 转换为 DataFrame 展示
print(pd.DataFrame(iqr_bounds).T) # .T 是转置，让列名变为索引，便于阅读

####  这里数据把follower_friend_ratio，tweets_per_day，aggressiveness，visibility（除数是参数的数据）所有0不去除之后计算IQR
#                                 Q1            Q3  ...   lower_bound   upper_bound
# followers_count         103.000000    753.000000  ...   -872.000000   1728.000000
# friends_count           167.000000    758.000000  ...   -719.500000   1644.500000
# statuses_count         1104.000000  17885.000000  ... -24067.500000  43056.500000
# retweet_count             0.000000      0.000000  ...      0.000000      0.000000
# favorite_count            0.000000      0.000000  ...      0.000000      0.000000
# mention_count             1.000000      2.000000  ...     -0.500000      3.500000
# url_count                 0.000000      1.000000  ...     -1.500000      2.500000
# text_length              97.000000    230.000000  ...   -102.500000    429.500000
# follower_friend_ratio     0.362000      1.395000  ...     -1.187500      2.944500
# active_hours              0.000000      6.750139  ...    -10.125208     16.875347
# tweets_per_day            0.000000      1.993556  ...     -2.990333      4.983889
# aggressiveness            0.000000      0.000701  ...     -0.001051      0.001752
# visibility                0.162857      0.490000  ...     -0.327857      0.980714
# reply_rate                0.000000      0.000000  ...      0.000000      0.000000
# quote_rate                0.000000      0.000000  ...      0.000000      0.000000

#----------------------------------两组数据区别在于是否去除0的4个参数----------------------------------------#

####  这里数据把follower_friend_ratio，tweets_per_day，aggressiveness，visibility（除数是参数的数据）所有0去除之后计算IQR
#                                 Q1            Q3  ...   lower_bound   upper_bound
# followers_count         103.000000    753.000000  ...   -872.000000   1728.000000
# friends_count           167.000000    758.000000  ...   -719.500000   1644.500000
# statuses_count         1104.000000  17885.000000  ... -24067.500000  43056.500000
# retweet_count             0.000000      0.000000  ...      0.000000      0.000000
# favorite_count            0.000000      0.000000  ...      0.000000      0.000000
# mention_count             1.000000      2.000000  ...     -0.500000      3.500000
# url_count                 0.000000      1.000000  ...     -1.500000      2.500000
# text_length              97.000000    230.000000  ...   -102.500000    429.500000
# follower_friend_ratio     0.368000      1.401000  ...     -1.181500      2.950500
# active_hours              0.000000      6.750139  ...    -10.125208     16.875347
# tweets_per_day            1.914563     14.917127  ...    -17.589284     34.420974
# aggressiveness            0.000682      0.007025  ...     -0.008832      0.016539
# visibility                0.164286      0.492857  ...     -0.328571      0.985714
# reply_rate                0.000000      0.000000  ...      0.000000      0.000000
# quote_rate                0.000000      0.000000  ...      0.000000      0.000000

'''

# -------- 根据所生成的上下限判断账号是否为异常 --------


# 初始化异常计数列，每个用户的异常特征数初始为 0
df['anomaly_feature_count'] = 0

# 遍历每个特征，对每一列检测是否越界，并累计异常数量
for feature in bounds_df.index:
    lower = bounds_df.loc[feature, 'lower']
    upper = bounds_df.loc[feature, 'upper']

    # 构造布尔掩码
    is_low = df[feature] < lower if pd.notna(lower) else False
    is_high = df[feature] > upper if pd.notna(upper) else False

    # 累加异常次数
    df['anomaly_feature_count'] += (is_low | is_high)

# 根据异常次数 >= 2 来判定最终 label
df['label'] = (df['anomaly_feature_count'] >= 2).astype(int)


# -------- 输出label的csv文件 --------

# 仅保留 user_id 和最终异常标签
output_df = df[['user_id', 'label']]

# 导出为 CSV 文件
output_df.to_csv("anomaly_labels.csv", index=False)

print("✅ 异常标注结果已保存为 anomaly_labels.csv")


#啊啊啊啊啊啊啊啊啊啊啊啊啊啊不对劲啊，怎么最后有403656个异常账号啊啊啊啊啊