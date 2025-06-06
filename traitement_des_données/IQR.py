### 用IQR来对数据进行可视化，获取相关合理的用于打标签的参数

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
# Q1            Q3  ...   lower_bound   upper_bound
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