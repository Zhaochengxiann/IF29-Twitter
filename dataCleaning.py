from pymongo import MongoClient
import numpy as np
import pandas as pd
from datetime import datetime

# 本地 MongoDB，默认端口 27017
client = MongoClient("mongodb://localhost:27017/")

# 连接数据库
db = client["IF29"]

# 连接集合
collection = db["IF29"]

# 查询一条数据
document = collection.find_one()

# 初始化结果字典
user_stats = {}

count = 0
# 遍历所有推文
for tweet in collection.find():
    try:
        count += 1
        if count % 10000 == 0:
            print(f"已处理 {count} 条 tweet...")
        user = tweet["user"]
        user_id = user["id"]
        followers_count = tweet["user"].get("followers_count", 0)
        created_at_str = tweet.get("created_at", "")
        created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")  # Twitter时间格式

        if user_id not in user_stats:
            user_stats[user_id] = {
                "user_id": user_id,
                "followers_count": followers_count,
                "friends_count": user.get("friends_count", 0),
                "statuses_count": user.get("statuses_count", 0),
                "retweet_count": 0,
                "favorite_count": 0,
                "mention_count": 0,
                "url_count": 0,
                "text_length": 0,
                "first_tweet_time": created_at,
                "last_tweet_time": created_at,
                "first_followers": followers_count,
                "last_followers": followers_count,
                "tweet_count": 1,
                "hashtag_count": 0,
                "reply_received_count": 0,
                "quote_count": 0
            }

        else:
            user_stats[user_id]["tweet_count"] += 1
            if created_at < user_stats[user_id]["first_tweet_time"]:
                user_stats[user_id]["first_tweet_time"] = created_at
                user_stats[user_id]["first_followers"] = followers_count
            if created_at > user_stats[user_id]["last_tweet_time"]:
                user_stats[user_id]["last_tweet_time"] = created_at
                user_stats[user_id]["last_followers"] = followers_count
            if tweet.get("reply_count", 0) > 0:
                user_stats[user_id]["reply_received_count"] += 1
            if tweet.get("quote_count", 0) > 0:
                user_stats[user_id]["quote_count"] += 1


    # 当前这条 tweet 的数据
        stats = user_stats[user_id]
        stats["retweet_count"] += tweet.get("retweet_count", 0)
        stats["favorite_count"] += tweet.get("favorite_count", 0)

        entities = tweet.get("entities", {})
        stats["mention_count"] += len(entities.get("user_mentions", []))
        stats["hashtag_count"] += len(entities.get("hashtags", []))
        stats["url_count"] += len(entities.get("urls", [])) + len(entities.get("media", []))

        stats["text_length"] += len(tweet.get("text", ""))

    except Exception as e:
        print("❌ 跳过异常数据:", e)

# 转换为 DataFrame
df = pd.DataFrame(user_stats.values())

# 粉丝/好友比列（避免除以 0）
df["follower_friend_ratio"] = df["followers_count"] / df["friends_count"].replace(0, pd.NA)
df["follower_friend_ratio"] = df["follower_friend_ratio"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

# 计算活跃小时数（最小为1，防止除以0）
df["active_hours"] = (df["last_tweet_time"] - df["first_tweet_time"]).dt.total_seconds() / 3600
# 计算每小时发推频率
df["tweets_per_hour"] = (df["tweet_count"] / df["active_hours"]).apply(lambda x: 0 if np.isinf(x) else x)
# 每天推文频率
df["tweets_per_day"] = (df["tweet_count"] / df["active_hours"] * 24).apply(lambda x: 0 if np.isinf(x) else x)
# 每小时新增关注频率
df["follower_per_hour"] = ((df["last_followers"] - df["first_followers"]) / df["active_hours"]).apply(lambda x: 0 if np.isinf(x) or pd.isna(x) else x)
# 攻击性
df["aggressiveness"] = ( df["tweets_per_hour"] + df["follower_per_hour"] ) / 140
# 能见度
df["visibility"] = ( df["mention_count"] * 11.4 + df["hashtag_count"] * 11.6 ) / 140
# 回复率
df["reply_rate"] = df["reply_received_count"] / df["tweet_count"]
# 引用率
df["quote_rate"] = df["quote_count"] / df["tweet_count"]

# 删除不需要导出的中间列
df.drop(columns=["first_tweet_time", "last_tweet_time","tweet_count","hashtag_count","reply_received_count","quote_count"], inplace=True)

# 导出为 CSV
df.to_csv("./data/cleaned_data_all.csv", index=False, encoding="utf-8-sig")

print("导出成功：cleaned_data_all.csv")