'''
此文件实现了标准化、降维、标签生成以及可视化PCA结果

输出文件：
- pca_anomaly_result.csv：包括用户ID、标准化特征、PCA坐标和异常标签
- 可视化弹窗：可视化部分使用随机抽样的5000条数据,避免窗口卡顿.
  可视化pca结果仅用于观察IF异常检测是否有效分类.
 预计执行时间：1min （可视化弹窗需要更多时间）

Ce fichier permet la normalisation, la réduction de la dimensionnalité, la génération d'étiquettes et la visualisation des résultats de l'ACP.

Fichier de sortie :
- pca_anomaly_result.csv : comprenant l'identifiant de l'utilisateur, les caractéristiques normalisées, les coordonnées de l'ACP et les étiquettes d'anomalie.
- La partie visualisation utilise un échantillon aléatoire de 5000 données pour éviter le décalage de fenêtre.
  Résultats pca visualisés sont uniquement utilisés pour vérifier si la détection d'anomalie IF est classée de manière efficace.
 Temps d'exécution estimé : 1min (Pop-ups de visualisation prennent plus de temps)

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# log1p (pour éviter que les valeurs extrêmes n'affectent l'ACP et l'IF)
X_log = X.apply(lambda col: np.log1p(col) if np.issubdtype(col.dtype, np.number) else col)

# standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduction PCA à deux dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Détection des anomalies : Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
anomaly_labels = iso.fit_predict(X_scaled)  # -1 est une anomalie, 1 est normal.

# Construire le DataFrame de sortie
df_scaled = pd.DataFrame(X_scaled, columns=[f'scaled_{f}' for f in features])
df_pca = pd.DataFrame(X_pca, columns=['pca_x', 'pca_y'])

df_result = pd.concat([df[['user_id']].reset_index(drop=True),
                       df_scaled.reset_index(drop=True),
                       df_pca.reset_index(drop=True)], axis=1)
df_result['anomaly'] = anomaly_labels

# CSV
df_result.to_csv("pca_anomaly_result.csv", index=False)
print("Le fichier de balises pca_anomaly_result.csv a été exporté avec succès. ")

# Visualisation : échantillon aléatoire de 5000 
df_viz = df_result.sample(n=5000, random_state=42)

plt.figure(figsize=(10, 7))
plt.title("Anomaly Detection via PCA + Isolation Forest (log1p + sampled 5000)", fontsize=14)

normal = df_viz[df_viz['anomaly'] == 1]
plt.scatter(normal['pca_x'], normal['pca_y'], c='blue', label='Normal',
            alpha=0.4, s=10)

anomalies = df_viz[df_viz['anomaly'] == -1]
plt.scatter(anomalies['pca_x'], anomalies['pca_y'], c='red', label='Anomaly',
            alpha=0.8, s=20, edgecolors='k')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.xlim(-5, 25)  
plt.ylim(-5, 25)
plt.tight_layout()
plt.show()

