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

Les données sont stockées dans une base MongoDB nommée **IF29**, collection **IF29**. Chaque document représente un tweet, contenant des métadonnées utilisateur.

### Script utilisé

Le script `dataCleaning.py` permet d’extraire des **statistiques agrégées par utilisateur** à partir des tweets.

**Principales étapes :**

- Chargement des données depuis MongoDB
- Agrégation des tweets par utilisateur (`user_id`)
- Calcul de diverses métriques comportementales
- Export d’un fichier CSV cleaned_data_all.csv pour la visualisation, la normalisation et la réduction de dimension (PCA, t-SNE), avant l’entraînement du modèle. 

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

## Idées pour l’entraînement et l’évaluation des modèles

🧠Nous n'avons pas encore commencé l'entraînement du modèle. Voici quelques idées préliminaires pour cette phase... ...

### Approche supervisée

#### Objectif :

- Apprendre à classifier les profils suspects via un modèle supervisé basé sur un jeu de données annoté (si disponible ou simulé).

#### Modèles envisagés :

- `Random Forest`
- `SVM`
- `Logistic Regression`
- `XGBoost`

#### Métriques d’évaluation :

- Accuracy
- Précision, rappel, F1-score
- Temps d’entraînement
- Matrice de confusion

------

### Approche non supervisée

#### Objectif :

- Identifier automatiquement des groupes d’utilisateurs similaires et repérer des comportements déviants sans étiquettes.

#### Méthodes envisagées :

- `K-Means`
- `DBSCAN`
- `Isolation Forest` (semi-supervisé)
- Réduction de dimension avec `PCA` ou `t-SNE`

#### Métriques :

- Silhouette Score
- Visualisation des clusters
- Évaluation manuelle par inspection des clusters suspects
