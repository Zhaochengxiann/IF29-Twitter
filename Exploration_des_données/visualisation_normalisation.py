#此文件实现了标准化、降维和标签的生成，运行此py文件后将在同路径下生成一个名为pca_anomaly_result.csv的标签csv文件
#Ce fichier permet la standardisation, la réduction de dimension et la génération d'étiquettes. Après avoir exécuté ce fichier py, un fichier csv d'étiquettes nommé pca_anomaly_result.csv sera généré dans le même chemin d'accès.

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

# Traitement des valeurs manquantes
X = df[features].fillna(0)

# standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduction PCA à deux dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Détection des anomalies : Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
anomaly_labels = iso.fit_predict(X_scaled)  # -1 est une anomalie, 1 est normal.

# Fusionner les résultats et générer le fichier de balises.
df_result = pd.DataFrame({
    'user_id': df['user_id'],
    'anomaly': anomaly_labels
})

df_result.to_csv("pca_anomaly_result.csv", index=False)
print("Le fichier de balises pca_anomaly_result.csv a été exporté avec succès. ")

#由于数据太大了，可视化部分只能通过取1000条数据实现（不然内存就炸了)
#可视化不会啊啊啊啊啊啊


