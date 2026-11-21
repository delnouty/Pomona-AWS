# Fruits! — Traitement Big Data sur le Cloud (AWS EMR)

Projet : « Réalisez un traitement dans un environnement Big Data sur le Cloud ».
Chaîne PySpark : images de fruits → features MobileNetV2 (poids broadcastés) → PCA → sortie sur S3.

## Livrables

- [ ] Notebook cloud (EMR) avec scripts PySpark exécutables : preprocessing + PCA
- [ ] Images du jeu de données initial et sortie de la PCA (CSV/parquet) sur S3
- [ ] Support de soutenance :
  - briques d'architecture cloud choisies et leur rôle
  - démarche de mise en place de l'environnement (EMR)
  - étapes de la chaîne PySpark

## Structure

| Dossier | Contenu |
|---|---|
| `src/pipeline.py` | Script PySpark (local et EMR) |
| `notebooks/00_alternant_original.ipynb` | Notebook de référence du mode opératoire P8 (intact) : extraction de features, sans PCA. Matériel de cours : local uniquement, non publié |
| `notebooks/P8_notebook.ipynb` | Notebook du manuel : chaîne complète en local (section 3) et sur EMR (section 4.10), avec broadcast et PCA ; c'est le livrable |
| `aws/bootstrap-emr.sh`, `aws/aws_steps.sh` | Fichiers du manuel : bootstrap EMR 7 et toutes les commandes AWS numérotées par étape |
| `aws/emr_create_cluster.sh`, `aws/emr_bootstrap.sh` | Mes scripts d'origine, basés sur EMR 6.3.0 (mode opératoire) ; à comparer avec ceux du manuel |
| `data/Test/` | Fruits-360 `Test` : 22 688 images, 131 classes (98 Mo), à uploader sur S3 |
| `data/Training/` | Fruits-360 `Training` : 67 692 images, 131 classes (299 Mo). Pas dans le périmètre de base ; sert au test de passage à l'échelle (3× plus de données) |
| `docs/P8_Textbook.pdf` | Manuel du projet : théorie, architecture, étapes AWS, soutenance. Local uniquement, non publié |
| `docs/mode_operatoire/img/` | Captures d'écran du mode opératoire (réutilisables pour la soutenance). Local uniquement, non publié |
| `docs/` | Architecture, retour critique, support de présentation (à venir) |

## Configuration de référence (mode opératoire P8)

| Élément | Valeur |
|---|---|
| Région | eu-west-3 (Paris) : conforme RGPD |
| EMR | emr-6.3.0 : Spark 3.1.1, JupyterHub 1.2.0, application TensorFlow 2.4.1 |
| Cluster | 1 maître + 2 cœurs, m5.xlarge |
| Jeu de données | dossier `Test` de Fruits-360 (22 688 images, 131 classes) sur S3 |
| Notebooks | persistés sur S3 (`jupyter-s3-conf`) |
| Résultat attendu | traitement des 22 688 images en un peu moins de 8 min |

Écarts par rapport au notebook de référence (ces choix suivent le manuel P8_Textbook) :

- ajout de la PCA, précédée d'un centrage (`StandardScaler(withMean=True, withStd=False)`) : le `transform` de Spark ne soustrait pas la moyenne
- k = plus petit nombre de composantes qui conserve 90 % de la variance, lu sur la courbe de variance cumulée
- `repartition(2 * sc.defaultParallelism)` : s'adapte à la taille du cluster
- lots Arrow de 256 images (`maxRecordsPerBatch`)
- modèle construit avec `include_top=False, pooling="avg"` : même vecteur de 1280 valeurs que `layers[-2]`

Le manuel utilise EMR 7.x (TensorFlow installé par le bootstrap) alors que la configuration ci-dessus vient du mode opératoire (EMR 6.3.0). À trancher par un essai réel sur AWS ; en cas d'échec d'EMR 7, garder 6.3.0 et le justifier dans le retour critique.

## Organisation S3

```text
s3://<bucket-eu>/
├── Test/<classe>/*.jpg    # images d'origine
├── Results/               # features MobileNetV2, 1280 valeurs (parquet)
├── Results_PCA/           # features réduites à k valeurs (parquet)
├── Results_PCA_variance/  # courbe de variance expliquée (CSV)
├── bootstrap-emr.sh       # actions d'amorçage EMR
├── jupyter/               # notebooks JupyterHub (persistance S3)
└── logs/                  # logs EMR
```

## Contraintes

- **RGPD** : bucket S3 et cluster EMR dans une région européenne (eu-west-3).
- **Budget < 10 €** : cluster allumé uniquement pour les tests et la démo ; développement en local sur un échantillon ; arrêt automatique après 1 h d'inactivité ; terminaison manuelle après chaque session.
- **Aucun secret dans le dépôt** : clés AWS via `aws configure` uniquement.

## Environnement local

Le développement local se fait sous WSL Ubuntu 24.04 (Python 3.12), comme dans le mode opératoire. Python 3.13 sous Windows n'est pas compatible avec PySpark et TensorFlow. Java 17 est requis par Spark.

```bash
# une seule fois (demande le mot de passe sudo)
sudo apt update && sudo apt install -y openjdk-17-jre-headless python3-venv python3-pip

# le venv reste dans le home WSL (plus rapide que /mnt/c)
python3 -m venv ~/venvs/fruits
source ~/venvs/fruits/bin/activate
cd /mnt/c/Users/DFILATOVA/Desktop/Fruits-AWS
pip install -r requirements.txt

# les images aussi sont lues bien plus vite depuis le système de fichiers Linux
mkdir -p ~/p8/data && cp -r data/Test ~/p8/data/Test

# --limit 3000 : tirage aléatoire de 3000 images dans Test
python src/pipeline.py --local --input ~/p8/data/Test --output ~/p8/output --limit 3000
```

Points mesurés pendant les tests locaux :

- **Au moins ~1500 images pour un test de la PCA.** Avec moins d'images que de features (1280), la matrice de covariance est de rang n-1 et la PCA de Spark a renvoyé des composantes fausses, différentes d'un lancement à l'autre (variance captée 59 % au lieu de 90 % pour le même k). Avec 3000 images, les résultats de Spark coïncident avec un calcul NumPy de référence, sur plusieurs ajustements. Le script affiche un `ATTENTION` dans les deux cas risqués.
- **`--cores 4` par défaut en local.** Chaque tâche lance un worker Python qui charge TensorFlow ; avec `local[*]` (22 cores ici) le processus Spark s'est arrêté sur 3000 images. Cause probable : mémoire, non confirmée par un journal système.
- **Données sur `/mnt/c` : lentes.** 300 images : 2 min 28 depuis `/mnt/c`, 41 s depuis `~/p8/data`.

## Utilisation sur EMR

Voir `aws/emr_create_cluster.sh` (chemins `s3://` en entrée et en sortie).

## État d'avancement

- [x] Squelette du projet, version initiale de `pipeline.py` (non testée)
- [x] Notebook de référence et captures importés
- [x] Java 17 et venv Python installés sous WSL
- [x] Jeu de données extrait (`data/Test`, `data/Training`)
- [x] `pipeline.py` aligné sur le manuel : centrage avant la PCA, k à 90 % de variance, dossiers `Results*`, `repartition(2 × cores)`
- [x] Test local sur 3000 images : 131 classes, k = 170 pour 90,0 % de variance, 56 s avec 4 cores ; PCA de Spark vérifiée contre NumPy
- [x] Notebook et scripts du manuel ajoutés (`notebooks/P8_notebook.ipynb`, `aws/bootstrap-emr.sh`, `aws/aws_steps.sh`)
- [ ] Exécuter la section 3 du notebook en local (voir ses réglages : `local[*]` et 150 images, cf. « Points mesurés »)
- [ ] Aligner `aws/` sur le manuel (EMR 7.x, bootstrap) après essai réel
- [ ] Bucket S3, rôles IAM, paire de clés, cluster EMR
- [ ] Exécution sur EMR, résultats sur S3, cluster terminé
- [ ] Retour critique et support de soutenance
