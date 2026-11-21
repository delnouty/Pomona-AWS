# Fruits! — Chaîne de traitement PySpark : images → features → PCA

Projet P8 : « Réalisez un traitement dans un environnement Big Data sur le Cloud ».

La start-up AgriTech *Fruits!* veut identifier des fruits à partir de photos. Ce dépôt contient les premières briques de la chaîne de traitement : transformer des images en vecteurs numériques compacts, avec un code écrit en PySpark pour pouvoir passer à l'échelle quand le volume de données augmentera. Aucun modèle n'est entraîné à ce stade.

## Chaîne de traitement

1. **Chargement** : Spark lit les images (`binaryFile`) ; l'étiquette est le nom du dossier parent.
2. **Extraction de features** : MobileNetV2 pré-entraîné sur ImageNet, sans la couche de classification, donne 1 280 valeurs par image (apprentissage par transfert, pas d'entraînement).
3. **Diffusion des poids (broadcast)** : les poids du modèle sont envoyés une fois à chaque exécuteur ; chacun reconstruit le réseau (`weights=None`) puis charge les poids reçus, sans rien télécharger.
4. **Réduction de dimension** : centrage des features, puis PCA de Spark ML. Le nombre de composantes k est le plus petit qui conserve 90 % de la variance.
5. **Écriture et validation** : résultats en Parquet, courbe de variance en CSV ; contrôle du nombre de lignes et de la taille des vecteurs avant et après la PCA.

## Résultats mesurés

Mesures locales (WSL Ubuntu, 4 cores) sur le jeu Fruits-360 :

| Exécution | Images | Classes | k pour 90 % de variance | Durée |
| --- | --- | --- | --- | --- |
| `src/pipeline.py`, tirage aléatoire dans `Test` | 3 000 | 131 | 170 | 56 s (extraction 26 s, PCA 15 s) |
| `notebooks/P8_notebook.ipynb`, section 3 | 1 492 | 10 | 44 | extraction 17 s, PCA 5 s |

Ces valeurs de k dépendent de l'échantillon : sur la totalité des 22 688 images de `Test`, elles seront différentes.

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
| `src/pipeline.py` | chaîne complète en un script (local ou cluster) : orchestration Spark et extraction des features |
| `src/pipeline_utils.py` | fonctions pures du pipeline (choix de k, décodage des images, contrôles, arguments), sans Spark ni TensorFlow ; testées unitairement |
| `notebooks/P8_notebook.ipynb` | notebook explicatif |
| `requirements.txt` | versions figées : PySpark 3.5.3, TensorFlow CPU 2.16.2, Keras 3.5.0, NumPy 1.26.4, pandas 2.2.3, PyArrow 17.0.0, Pillow 10.4.0 |
| `aws/` | scripts de déploiement sur AWS EMR, fournis à titre de référence |
| `.gitignore`, `.gitattributes` | exclusions (données, résultats, clés, environnements) ; fins de ligne LF pour les scripts `.sh` |
| `.pre-commit-config.yaml`, `ruff.toml`, `requirements-dev.txt` | contrôles locaux à chaque commit (voir « Contribuer ») |
| `.github/workflows/ci.yml` | intégration continue sur GitHub Actions |
| `tests/`, `pytest.ini`, `requirements-test.txt` | tests unitaires de `src/pipeline_utils.py` et leurs dépendances (voir « Contribuer ») |

## Périmètre et état

- [x] Chaîne PySpark complète : chargement, extraction MobileNetV2, broadcast des poids, centrage, PCA, validation
- [x] Exécution et vérification en local (voir « Résultats mesurés »)
- [ ] Exécution sur un cluster cloud : approche à définir. Les scripts de `aws/` (EMR, S3) n'ont **pas** été exécutés.
- [ ] Retour critique sur la solution
- [ ] Support de présentation

Contraintes du sujet : en cas de déploiement cloud, serveurs situés dans l'Union européenne (RGPD) et coût maîtrisé (cluster allumé uniquement pour les tests). Les images Fruits-360 ne contiennent pas de données personnelles ; une exécution locale ne transfère aucune donnée.

## Contribuer

### Branches

- `main` : versions stables. Pas de commit direct : un hook l'empêche en local.
- `develop` : intégration du travail en cours.
- Pour une modification : créer une branche depuis `develop` (par exemple `feature/ma-modification`), puis ouvrir une pull request vers `develop`.

### Installer les contrôles locaux (pre-commit)

À faire une fois, **depuis l'environnement dans lequel vous faites vos commits** : le hook mémorise l'interpréteur Python utilisé à l'installation.

Linux ou WSL :

```bash
python3 -m venv ~/venvs/dev && source ~/venvs/dev/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

Windows (PowerShell) :

```powershell
py -3.12 -m pip install -r requirements-dev.txt
py -3.12 -m pre_commit install
```

Pour contrôler tout le dépôt à la main : `pre-commit run --all-files` (la première exécution télécharge les environnements des hooks, donc nécessite internet).

### Lancer les tests unitaires

Les tests n'ont besoin ni de Spark ni de TensorFlow : un environnement léger suffit.

```bash
python3 -m venv ~/venvs/test && source ~/venvs/test/bin/activate
pip install -r requirements-test.txt
python -m pytest
```

Chaque test vérifie une seule fonction de `src/pipeline_utils.py`, de façon isolée : aucun fichier, aucune session Spark ; les images sont fabriquées en mémoire.

### Ce qui est contrôlé

| Où | Contrôles |
| --- | --- |
| Hooks locaux, à chaque commit | pas de commit sur `main` ; fichiers de plus de 1 Mo ; clés privées ; marqueurs de conflit ; YAML valide ; espaces en fin de ligne et fin de fichier (notebooks exclus) ; lint et format avec `ruff` |
| CI GitHub Actions, à chaque push et pull request vers `develop` ou `main` | les mêmes hooks (sauf l'interdiction de commit sur `main`) ; syntaxe de `src/pipeline.py` ; notebook en JSON valide ; tests unitaires ; recherche de secrets avec `gitleaks` |

**Limite actuelle :** les tests unitaires couvrent les fonctions pures de `src/pipeline_utils.py`. Le code qui pilote Spark et TensorFlow dans `src/pipeline.py` n'est couvert par aucun test automatique ; il a été vérifié à la main par des exécutions locales.
