
### Utilisez l'IQR（Interquartile Range） pour visualiser les données et obtenir des paramètres pertinents et raisonnables pour l'étiquetage.

import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv("../dataset/cleaned_data_all.csv")

# Remplacer les valeurs manquantes par 0.
df["follower_friend_ratio"] = df["follower_friend_ratio"].fillna(0)

features = [
    'followers_count','friends_count','statuses_count', 'mention_count', 'url_count', 'text_length',
    'follower_friend_ratio', 'active_hours', 'tweets_per_day','aggressiveness', 'visibility'
]

# S'assurer que tous les champs de la section « features » se trouvent dans « df_clean ».
X = df[features]

# Vérifiez le nombre de valeurs uniques dans chaque colonne (si d'autres colonnes sont presque constantes, vous pouvez également envisager de les supprimer).
print(X.nunique())


X = df[features]

# Méthode de calcul de l'IQR bilatéral
def calculate_iqr_bounds(column_data):
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Méthode de calcul de l'IQR unilatéral
def calculate_high_iqr_bound(column_data):
    Q3 = column_data.quantile(0.75)
    Q1 = column_data.quantile(0.25)
    IQR = Q3 - Q1
    return Q3 + 1.5 * IQR

# Utilisation de la méthode des centiles pour calculer les limites supérieure et inférieure
def calculate_percentile_bounds(column_data, lower_percentile=1, upper_percentile=99):
    return column_data.quantile(lower_percentile/100), column_data.quantile(upper_percentile/100)

# Enregistrer la méthode de détection des anomalies et les seuils correspondant à chaque caractéristique.
bounds = {}

# -------- Effectuer une évaluation automatique de chaque colonne de caractéristiques et calculer les limites correspondantes pour les valeurs aberrantes. --------
for col in features:
    uniq = X[col].nunique()       # Nombre de valeurs uniques
    total = len(X[col])           # Nombre total d'échantillons
    
    # Cas 1 : s'il s'agit d'une caractéristique de type ratio ou d'un indicateur présentant un biais droit évident → utiliser l'IQR unilatéral (détecter uniquement les valeurs anormales élevées)
    if 'ratio' in col or col in ['tweets_per_day', 'aggressiveness']:
        method = 'IQR-High'
        upper = calculate_high_iqr_bound(X[col])
        bounds[col] = {'method': method, 'lower': None, 'upper': upper}
    
    # Cas 2 : Si les valeurs uniques sont peu nombreuses et que la taille de l'échantillon est importante → Utiliser la méthode des percentiles pour contrôler la proportion d'anomalies.
    elif uniq < 1000 and total > 100000:
        method = 'Percentile'
        lb, ub = calculate_percentile_bounds(X[col], 1, 99)  
        bounds[col] = {'method': method, 'lower': lb, 'upper': ub}
    
    # Cas 3 : utilisation par défaut de la méthode IQR bilatérale pour détecter les anomalies (applicable à la plupart des variables continues)
    else:
        method = 'IQR-Both'
        lb, ub = calculate_iqr_bounds(X[col])
        bounds[col] = {'method': method, 'lower': lb, 'upper': ub}

# -------- Méthode de détection des valeurs aberrantes et seuil pour toutes les caractéristiques --------

# Convertir les résultats en DataFrame
bounds_df = pd.DataFrame.from_dict(bounds, orient='index')

# Réinitialiser le nom de l'index en feature (à l'origine, c'était le nom de la colonne).
bounds_df.index.name = 'feature'

# Afficher les résultats
print(bounds_df)


# -------- Déterminer si le compte est anormal en fonction des limites supérieure et inférieure générées. --------

# Initialiser la série de comptage des anomalies, le nombre d'anomalies caractéristiques de chaque utilisateur étant initialement égal à 0.
df['anomaly_feature_count'] = 0

# Parcourir chaque caractéristique, vérifier si chaque colonne dépasse les limites et cumuler le nombre d'anomalies.
for feature in bounds_df.index:
    lower = bounds_df.loc[feature, 'lower']
    upper = bounds_df.loc[feature, 'upper']

    # Construire un masque booléen
    is_low = df[feature] < lower if pd.notna(lower) else False
    is_high = df[feature] > upper if pd.notna(upper) else False

    # Nombre cumulé d'anomalies
    df['anomaly_feature_count'] += (is_low | is_high)

# Déterminer le label final en fonction du nombre d'anomalies >= 2.
df['label'] = (df['anomaly_feature_count'] >= 2).astype(int)


# -------- Fichier CSV contenant les étiquettes --------

df.to_csv("cleaned_data_with_anomaly_label.csv", index=False)

print("Le fichier contenant toutes les propriétés et les étiquettes d'anomalie a été enregistré sous le nom cleaned_data_with_anomaly_label.csv.")

# -------- Afficher le taux d'anomalies --------
total_count = len(df)
anomaly_count = df['label'].sum()
anomaly_ratio = anomaly_count / total_count

print(f"Proportion d'utilisateurs anormaux : {anomaly_ratio:.2%}")