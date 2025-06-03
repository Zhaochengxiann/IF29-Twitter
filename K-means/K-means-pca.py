from sklearn.cluster import KMeans
import pandas as pd

# 载入数据
df = pd.read_csv("../Exploration_des_données/pca_anomaly_result.csv")

# 提取标准化特征用于聚类
feature_cols = [col for col in df.columns if col.startswith('scaled_')]
X_cluster = df[feature_cols]

# 应用 K-Means（假设我们分成 5 个簇，可调整）
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_cluster)

# 将聚类结果加入到原始 DataFrame
df['cluster'] = cluster_labels

# 保存含聚类标签的结果
df.to_csv("kmeans_cluster_result.csv", index=False)
print("K-Means 聚类已完成，结果已保存至 kmeans_cluster_result.csv")

# 查看每个簇的用户数量
print(df['cluster'].value_counts())

# 打印 cluster 2、3、4 中的 user_id
anomalous_clusters = [2, 3, 4]
df_anomalies = df[df['cluster'].isin(anomalous_clusters)]

for cluster_id in anomalous_clusters:
    print(f"\n--- Cluster {cluster_id} ---")
    print(df[df['cluster'] == cluster_id]['user_id'].tolist())

