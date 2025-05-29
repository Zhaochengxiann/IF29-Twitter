import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../dataset/cleaned_data_all.csv")


features = [
    'followers_count', 'friends_count', 'statuses_count',
    'mention_count', 'url_count', 'text_length',
    'follower_friend_ratio', 'active_hours', 'tweets_per_day',
    'aggressiveness', 'visibility'
]

# 处理缺失值
X = df[features].fillna(0)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 降到二维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 异常检测：Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
anomaly_labels = iso.fit_predict(X_scaled)  # -1 是异常，1 是正常

# 合并结果，并输出标签文件
df_result = pd.DataFrame({
    'user_id': df['user_id'],
    'anomaly': anomaly_labels
})


df_result.to_csv("pca_anomaly_result.csv", index=False)
print("Le fichier de balises pca_anomaly_result.csv a été exporté avec succès. ")

#由于数据太大了，可视化部分只能通过取1000条数据实现（不然内存就炸了)
#可视化不会啊啊啊啊啊啊


