import pandas as pd

# 读取数据
df = pd.read_csv("../dataset/cleaned_data_all.csv")

# 填补空值
df = df.fillna(0)

# 自定义规则函数
def rule_based_anomaly(row):
    score = 0

    # 规则1：粉丝关注比太小
    if row['follower_friend_ratio'] < 0.05:
        score += 1

    # 规则2：发推量很大但无人关注
    if row['visibility'] < 1 and row['statuses_count'] > 10000:
        score += 1

    # 规则3：日均发推 > 50
    if row['tweets_per_day'] > 200:
        score += 1

    # 规则4：攻击性言论多
    if row['aggressiveness'] > 4:
        score += 1

    # 规则5：过于短的文本
    if row['text_length'] < 20:
        score += 1

    # 规则6：异常活跃（全天或极少）
    if row['active_hours'] < 3 or row['active_hours'] > 20:
        score += 1

    # 如果异常行为 ≥ 2 条，就认为是异常账号
    return 1 if score >= 2 else 0

# 应用规则
df['anomaly'] = df.apply(rule_based_anomaly, axis=1)

# 导出结果
df.to_csv("label_result.csv", index=False)

