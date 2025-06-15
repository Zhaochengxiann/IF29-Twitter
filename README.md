# IF29 - Classification de profils Twitter

## Contexte du projet

Ce projet a pour objectif d’identifier des profils utilisateurs suspects sur Twitter (bots, spammeurs, agents de désinformation, etc.) en comparant deux approches : **l’apprentissage supervisé** et **l’apprentissage non supervisé**.  

Nous avons utilisé une base de données MongoDB contenant des tweets en format JSON issus de la base **Tweet_Worldcup**. Ces tweets sont ensuite transformés en un fichier CSV structuré avec des variables utilisateurs agrégées.

---

## Configuration de l’environnement

### Prérequis

- Python 3.8 ou supérieur  
- MongoDB installé et en cours d’exécution (localhost:27017)  

### Installation des dépendances

```bash
pip install pandas numpy pymongo
```

## Traitement des données

### Source des données

Les données brutes sont collectées depuis **MongoDB** et transformées en un DataFrame utilisateur.

Exemples de caractéristiques calculées :

- `follower_friend_ratio`
- `tweets_par_jour`
- `visibility` (via mentions + hashtags)
- `aggressiveness` (activité combinée + croissance)
- `reply_rate`, `quote_rate`

➡️ Export final : `cleaned_data_all.csv`

Pour la version supervisée, un **label binaire** (`label`) est attribué selon des critères heuristiques inspirés de la littérature (SPOT, Botometer, etc.).

➡️ Export : `cleaned_data_with_anomaly_label.csv`

### Variables extraites

| Variable                | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `user_id`               | Identifiant unique de l’utilisateur Twitter                  |
| `followers_count`       | Nombre d’abonnés de l’utilisateur                            |
| `friends_count`         | Nombre d’abonnements de l’utilisateur                        |
| `statuses_count`        | Nombre total de tweets postés (statuts)                      |
| `retweet_count`         | Nombre total de retweets générés par l’utilisateur           |
| `favorite_count`        | Nombre total de "likes" reçus                                |
| `mention_count`         | Nombre total de mentions `@` dans ses tweets                 |
| `url_count`             | Nombre total d’URLs ou de médias insérés dans ses tweets     |
| `text_length`           | Somme des longueurs des tweets (en nombre de caractères)     |
| `follower_friend_ratio` | Ratio abonnés/abonnements *(indicatif d’influence ou de comportement bot)* |
| `active_hours`          | Durée d’activité entre le premier et le dernier tweet (en heures) |
| `tweets_per_day`        | Fréquence moyenne de publication (tweets/jour)               |
| `aggressiveness`        | Indicateur combiné d’activité et de croissance (tweets + followers) |
| `visibility`            | Score basé sur les mentions et hashtags *(visibilité potentielle)* |
| `reply_rate`            | Proportion de tweets ayant reçu une réponse                  |
| `quote_rate`            | Proportion de tweets ayant été cités (retweet avec commentaire) |

------

## Méthode non-supervisée : K-Means + PCA

- **Réduction de dimension** : PCA à 7 composantes (≥ 90% de variance)
- **Clustering** : `MiniBatchKMeans` sur 100 000 profils
- **Choix de k** : basé sur le Silhouette Score → **k = 2**
- **Visualisation** : UMAP + export statique `Kmeans_visu.png`

Résultat :

- Cluster 0 : 22,8 % des utilisateurs
- Cluster 1 : 77,2 % des utilisateurs

Les clusters montrent une séparation claire, et les comportements du cluster minoritaire suggèrent des profils suspects.


## Méthode supervisée : SVM

- PCA à 6 composantes principales (variance ≥ 80 %)
- Labelisation binaire : `0 = Non suspect`, `1 = Suspect`
- Modèle : `SVM` (kernel RBF)
- Dataset : 700 000 utilisateurs

### Résultats sur l'ensemble de test :

| Métrique  | Score   |
| --------- | ------- |
| Accuracy  | 99.71 % |
| Précision | 99.34 % |
| Rappel    | 99.98 % |
| F1-Score  | 99.66 % |

➡️ Visualisation : Matrice de confusion annotée



## Lancer les scripts

```python
# Nettoyage des données
python traitement_des_données/dataCleaning.py

# Génération des labels (version supervisée)
python traitement_des_données/label_final.py
```
