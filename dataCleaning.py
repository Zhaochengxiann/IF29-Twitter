from pymongo import MongoClient
import numpy as np
import pandas as pd
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["IF29"]
collection = db["test"]
document = collection.find_one()
user_stats = {}

count = 0
for tweet in collection.find():
    try:
        count += 1
        if count % 10000 == 0:
            print(f" {count} tweet transformés...")
        user = tweet["user"]
        user_id = user["id"]
        followers_count = tweet["user"].get("followers_count", 0)
        friends_count = tweet["user"].get("friends_count",0)
        created_at_str = tweet.get("created_at", "")
        created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")  

        if user_id not in user_stats:
            user_stats[user_id] = {
                "user_id": user_id,
                "followers_count": followers_count,
                "friends_count": friends_count,
                "statuses_count": user.get("statuses_count", 0),
                "retweet_count": 0,
                "favorite_count": 0,
                "mention_count": 0,
                "url_count": 0,
                "text_length": 0,
                "first_tweet_time": created_at,
                "last_tweet_time": created_at,
                "first_friends": friends_count,
                "last_friends": friends_count,
                "tweet_count": 1,
                "hashtag_count": 0,
                "reply_received_count": 0,
                "quote_count": 0
            }

        else:
            user_stats[user_id]["tweet_count"] += 1
            if created_at < user_stats[user_id]["first_tweet_time"]:
                user_stats[user_id]["first_tweet_time"] = created_at
                user_stats[user_id]["first_friends"] = friends_count
            if created_at > user_stats[user_id]["last_tweet_time"]:
                user_stats[user_id]["last_tweet_time"] = created_at
                user_stats[user_id]["last_friends"] = friends_count
            if tweet.get("reply_count", 0) > 0:
                user_stats[user_id]["reply_received_count"] += 1
            if tweet.get("quote_count", 0) > 0:
                user_stats[user_id]["quote_count"] += 1

        stats = user_stats[user_id]
        stats["retweet_count"] += tweet.get("retweet_count", 0)
        stats["favorite_count"] += tweet.get("favorite_count", 0)

        entities = tweet.get("entities", {})
        stats["mention_count"] += len(entities.get("user_mentions", []))
        stats["hashtag_count"] += len(entities.get("hashtags", []))
        stats["url_count"] += len(entities.get("urls", [])) + len(entities.get("media", []))

        stats["text_length"] += len(tweet.get("text", ""))

    except Exception as e:
        print("❌ Sauter les données d'exception:", e)

#DataFrame
df = pd.DataFrame(user_stats.values())

# follower_friend_ratio
df["follower_friend_ratio"] = df["followers_count"] / df["friends_count"].replace(0, pd.NA)
df["follower_friend_ratio"] = df["follower_friend_ratio"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

# heures_actives
df["active_hours"] = (df["last_tweet_time"] - df["first_tweet_time"]).dt.total_seconds() / 3600
# tweets_par_heure
df["tweets_per_hour"] = (df["tweet_count"] / df["active_hours"]).apply(lambda x: 0 if np.isinf(x) else x)
# "tweets_par_heure
df["tweets_par_jour"] = (df["tweet_count"] / df["active_hours"] * 24).apply(lambda x: 0 if np.isinf(x) else x)
# friends_per_hour
df["amis_par_heure"] = ((df["last_friends"] - df["first_friends"]).abs() / df["active_hours"]).apply(lambda x: 0 if np.isinf(x) or pd.isna(x) else x)
# agressivité  - Les nombres maximums d'API sont dérivés des journaux Twitter POST Endpoint Rate Limit pour 2018.
df["aggressiveness"] = ( df["tweets_per_hour"] + df["friends_per_hour"] ) / 140
# visibilité  -En utilisant la formule de SPOT 1.0, basée sur les changements 2018 de la limite de caractères du canal Twitter officiel
df["visibility"] = ( df["mention_count"] * 11.4 + df["hashtag_count"] * 11.6 ) / 280
# taux de réponse
df["reply_rate"] = df["reply_received_count"] / df["tweet_count"]
# taux de citation
df["quote_rate"] = df["quote_count"] / df["tweet_count"]

df.drop(columns=["first_tweet_time", "last_tweet_time","tweet_count","hashtag_count","reply_received_count","quote_count"], inplace=True)

df.to_csv("./dataset/cleaned_data_all_test.csv", index=False, encoding="utf-8-sig")

print("Exportation réussie：cleaned_data_all.csv")
