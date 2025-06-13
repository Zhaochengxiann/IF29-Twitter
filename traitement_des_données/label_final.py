import pandas as pd

# Charger les données
df = pd.read_csv(r"..\dataset\cleaned_data_all.csv")

# Remplacer les valeurs manquantes (essentiel pour ratio)
df["follower_friend_ratio"] = df["follower_friend_ratio"].fillna(0)

# -------- Critères fondés sur littérature SPOT, Botometer, Twitter API -------- #

# 1. follower_friend_ratio trop bas : typique d'un bot qui suit en masse mais peu suivi
critere_ratio = df["follower_friend_ratio"] < 0.1

# 2. active_hours très court (compte qui ne dure que quelques heures)
critere_duree_active = df["active_hours"] < 24  # moins d'un jour d'activité

# 3. tweets_per_day très élevé (activité automatisée)
critere_frequence = df["tweets_per_day"] > 50

# 4. aggressiveness anormale (hyperactivité combinée tweets + follows)
critere_aggressivite = df["aggressiveness"] > 1

# 5. visibilité très élevée (usage abusif de hashtags/mentions)
critere_visibilite = df["visibility"] > 5

# -------- Attribution du label -------- #
# Si l'utilisateur satisfait à au moins un de ces critères, on le considère suspect
df["label"] = (
    critere_ratio |
    critere_duree_active |
    critere_frequence |
    critere_aggressivite |
    critere_visibilite
).astype(int)

# -------- Export final -------- #
df.to_csv("..\dataset\cleaned_data_with_validated_label.csv", index=False)
print("Fichier exporté : cleaned_data_with_validated_label.csv")

# -------- Optionnel : Résumé rapide -------- #
nb_total = len(df)
nb_suspects = df["label"].sum()
print(f"\nRésumé : {nb_suspects} profils suspects détectés sur {nb_total} (soit {nb_suspects/nb_total:.2%})")
