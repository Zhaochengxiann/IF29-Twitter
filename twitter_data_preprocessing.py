import os
import json
import pandas as pd

# ==== 1. 读取 JSON 文件 ====
json_file_path = "raw0.json"  # 请根据你的路径修改
with open(json_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

data = [json.loads(line) for line in lines]

# 扁平化 JSON 数据
df = pd.json_normalize(data)


# ==== 2. 特征提取函数 ====
def extract_features(df):
    df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')

    # 提取用户和内容特征
    df["user_screen_name"] = df["user.screen_name"]
    df["followers_count"] = df["user.followers_count"]
    df["friends_count"] = df["user.friends_count"]
    df["statuses_count"] = df["user.statuses_count"]
    df["text"] = df["text"]
    df["text_length"] = df["text"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df["lang"] = df["lang"]
    df["retweet_count"] = df["retweet_count"]
    df["favorite_count"] = df["favorite_count"]

    # 标签和URL数量
    df["num_hashtags"] = df["entities.hashtags"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["num_urls"] = df["entities.urls"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # 粉丝/关注比
    df["followers_to_friends_ratio"] = df["followers_count"] / (df["friends_count"] + 1)

    # 账户创建天数
    df["account_created_at"] = pd.to_datetime(df["user.created_at"], errors='coerce').dt.tz_localize(None)
    now_naive = pd.Timestamp.now().tz_localize(None)
    df["account_age_days"] = (now_naive - df["account_created_at"]).dt.days

    # 选择要保留的字段
    selected_columns = [
        "user_screen_name", "followers_count", "friends_count", "statuses_count",
        "text", "text_length", "lang", "retweet_count", "favorite_count",
        "num_hashtags", "num_urls", "followers_to_friends_ratio",
        "account_age_days", "created_at"
    ]
    return df[selected_columns]


# ==== 3. 执行特征提取 ====
df_cleaned = extract_features(df)

# ==== 4. 保存结果 ====
output_file = "twitter_cleaned.csv"
df_cleaned.to_csv(output_file, index=False)
print(f"✅ 数据清洗完成，已保存至：{output_file}")


