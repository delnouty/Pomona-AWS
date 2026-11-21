# Fruits! — Chaîne de traitement PySpark : images → features → PCA

Projet P8 : « Réalisez un traitement dans un environnement Big Data sur le Cloud ».

La start-up AgriTech *Fruits!* veut identifier des fruits à partir de photos. Ce dépôt contient les premières briques de la chaîne de traitement : transformer des images en vecteurs numériques compacts, avec un code écrit en PySpark pour pouvoir passer à l'échelle quand le volume de données augmentera. Aucun modèle n'est entraîné à ce stade.

## Chaîne de traitement

1. **Chargement** : Spark lit les images (`binaryFile`) ; l'étiquette est le nom du dossier parent.
2. **Extraction de features** : MobileNetV2 pré-entraîné sur ImageNet, sans la couche de classification, donne 1 280 valeurs par image (apprentissage par transfert, pas d'entraînement).
3. **Diffusion des poids (broadcast)** : les poids du modèle sont envoyés une fois à chaque exécuteur ; chacun reconstruit le réseau (`weights=None`) puis charge les poids reçus, sans rien télécharger.
4. **Réduction de dimension** : centrage des features, puis PCA de Spark ML. Le nombre de composantes k est le plus petit qui conserve 90 % de la variance.
5. **Écriture et validation** : résultats en Parquet, courbe de variance en CSV ; contrôle du nombre de lignes et de la taille des vecteurs avant et après la PCA.

Le centrage est nécessaire : le `transform` de la PCA de Spark ne soustrait pas la moyenne, donc sans lui les coordonnées enregistrées seraient décalées d'un vecteur constant.

## Résultats mesurés

Mesures locales (WSL Ubuntu, 4 cores) sur le jeu Fruits-360 :

| Exécution | Images | Classes | k pour 90 % de variance | Durée |
| --- | --- | --- | --- | --- |
| `src/pipeline.py`, tirage aléatoire dans `Test` | 3 000 | 131 | 170 | 56 s (extraction 26 s, PCA 15 s) |
| `notebooks/P8_notebook.ipynb`, section 3 | 1 492 | 10 | 44 | extraction 17 s, PCA 5 s |

Ces valeurs de k dépendent de l'échantillon : sur la totalité des 22 688 images de `Test`, elles seront différentes.

## Points d'attention mesurés

- **Utiliser plus d'images que de features (au moins ~1 500) pour tester la PCA.** Avec 300 images (moins que les 1 280 features), la PCA de Spark a renvoyé des composantes fausses et différentes d'un lancement à l'autre : pour le même k = 95, la variance captée valait 90,1 % deux fois et 58,9 % une fois ; pour k = 50, 44,6 % au lieu de 79,7 %. Avec 3 000 images, Spark coïncide avec un calcul NumPy de référence sur tous les ajustements répétés. La cause exacte n'est pas établie ; le cas coïncide avec un nombre d'images inférieur au nombre de features. Le script affiche un `ATTENTION` quand l'échantillon est trop petit ou quand la courbe et le modèle ajusté divergent.
- **Limiter les cores en local (`--cores 4` par défaut).** Chaque tâche lance un worker Python qui charge TensorFlow. Avec `local[*]` sur 22 cores, le processus Spark s'est arrêté sur 3 000 images. Cause probable : la mémoire, non confirmée par un journal système.
- **Lire les images depuis le système de fichiers Linux.** Sous WSL, 300 images : 2 min 28 depuis le disque Windows (`/mnt/c`), 41 s depuis le disque Linux.

## Lancer en local

Prérequis : Linux ou WSL, Java 17, Python 3.10 à 3.12 (testé avec 3.12).

```bash
python3 -m venv ~/venvs/fruits
source ~/venvs/fruits/bin/activate
pip install -r requirements.txt
```

Données : le jeu [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) (images 100 × 100, 131 classes ; dossier `Test` : 22 688 images). Il n'est pas versionné ; placer le dossier `Test` hors du dépôt, par exemple dans `~/p8/data/Test`.

```bash
python src/pipeline.py --local --input ~/p8/data/Test --output ~/p8/output --limit 3000
```

| Option | Rôle |
| --- | --- |
| `--local` | session Spark locale ; à omettre sur un cluster |
| `--cores` | cores de la session locale (défaut 4) |
| `--limit` | tire au hasard N images (graine fixe) ; 0 = toutes |
| `--threshold` | part de variance à conserver pour choisir k (défaut 0,90) |
| `--n-components` | impose k au lieu de le déduire du seuil |
| `--k-max` | nombre maximal de composantes étudiées pour la courbe (défaut 300) |
| `--partitions` | partitions avant le réseau (défaut : 2 par core) |

Sorties dans le dossier `--output` : `Results/` (features, 1 280 valeurs), `Results_PCA/` (features réduites à k valeurs) et `Results_PCA_variance/` (courbe de variance expliquée, CSV).

## Notebook

`notebooks/P8_notebook.ipynb` déroule la même chaîne, étape par étape, avec les explications (en français).

- **Section 3** : exécution locale. Elle lit un échantillon `~/p8/data/Test_local` (quelques variétés de `Test`, plus de 1 280 images au total) et écrit dans `~/p8/output`. À lancer avec un noyau Python qui dispose des paquets de `requirements.txt` ; le nombre de cores se règle avec `N_CORES`.
- **Section 4** : déploiement sur un cluster cloud. Non exécutée dans ce dépôt (voir « Périmètre et état »).

## Contenu du dépôt

| Fichier | Rôle |
| --- | --- |
| `src/pipeline.py` | chaîne complète en un script (local ou cluster) |
| `notebooks/P8_notebook.ipynb` | notebook explicatif |
| `requirements.txt` | versions figées : PySpark 3.5.3, TensorFlow CPU 2.16.2, Keras 3.5.0, NumPy 1.26.4, pandas 2.2.3, PyArrow 17.0.0, Pillow 10.4.0 |
| `aws/` | scripts de déploiement sur AWS EMR, fournis à titre de référence |
| `.gitignore`, `.gitattributes` | exclusions (données, résultats, clés, environnements) ; fins de ligne LF pour les scripts `.sh` |

## Périmètre et état

- [x] Chaîne PySpark complète : chargement, extraction MobileNetV2, broadcast des poids, centrage, PCA, validation
- [x] Exécution et vérification en local (voir « Résultats mesurés » et « Points d'attention »)
- [ ] Exécution sur un cluster cloud : approche à définir. Les scripts de `aws/` (EMR, S3) n'ont **pas** été exécutés.
- [ ] Retour critique sur la solution
- [ ] Support de présentation

Contraintes du sujet : en cas de déploiement cloud, serveurs situés dans l'Union européenne (RGPD) et coût maîtrisé (cluster allumé uniquement pour les tests). Les images Fruits-360 ne contiennent pas de données personnelles ; une exécution locale ne transfère aucune donnée.

## Branches

- `develop` : travail en cours.
- `main` : versions stables ; pas de poussée directe.
