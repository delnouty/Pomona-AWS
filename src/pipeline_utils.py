"""Fonctions pures de la chaîne Fruits! : sans Spark, sans TensorFlow, sans accès au disque.

Elles sont séparées de pipeline.py pour pouvoir être testées unitairement (tests/).
"""

import argparse
import io

import numpy as np
import pandas as pd
from PIL import Image

IMG_SIZE = (224, 224)  # taille d'entrée attendue par MobileNetV2
N_FEATURES = 1280  # taille du vecteur de sortie de MobileNetV2


def build_parser(description=None):
    """Arguments de la ligne de commande de pipeline.py."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="dossier des images (local ou s3://)")
    parser.add_argument("--output", required=True, help="dossier de sortie (local ou s3://)")
    parser.add_argument(
        "--local", action="store_true", help="session Spark locale ; à omettre sur EMR"
    )
    parser.add_argument(
        "--cores", type=int, default=4, help="cores de la session locale (avec --local uniquement)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="limite le nombre d'images (tests locaux), 0 = toutes"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90, help="part de variance à conserver pour choisir k"
    )
    parser.add_argument(
        "--n-components", type=int, default=None, help="impose k au lieu de le déduire du seuil"
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=300,
        help="nombre maximal de composantes étudiées pour la courbe",
    )
    parser.add_argument(
        "--partitions", type=int, default=None, help="partitions avant le CNN (défaut : 2 par core)"
    )
    return parser


def output_paths(output):
    """Dossiers de sortie, sous le dossier de sortie donné (local ou s3://)."""
    return {
        "features": f"{output}/Results",  # features (1280 valeurs)
        "pca": f"{output}/Results_PCA",  # features après PCA
        "variance": f"{output}/Results_PCA_variance",  # variance expliquée (CSV)
    }


def max_components(k_max, n_rows, n_features=N_FEATURES):
    """Nombre maximal de composantes étudiées : au plus n-1 portent de la variance."""
    return int(min(k_max, n_rows - 1, n_features))


def too_few_images(n_rows, n_features=N_FEATURES):
    """Vrai si la PCA risque d'être peu fiable : pas plus d'images que de features."""
    return n_rows <= n_features


def variance_table(explained):
    """Tableau k, variance expliquée, variance cumulée (k commence à 1)."""
    explained = np.asarray(explained, dtype=float)
    return pd.DataFrame(
        {
            "k": np.arange(1, len(explained) + 1),
            "explained": explained,
            "cumulative": np.cumsum(explained),
        }
    )


def threshold_reached(cumulative, threshold):
    """Vrai si la variance cumulée atteint le seuil pour au moins un k."""
    return bool((np.asarray(cumulative) >= threshold).any())


def select_k(cumulative, threshold, n_components=None):
    """Nombre de composantes à garder.

    - n_components imposé : on le prend, s'il ne dépasse pas le nombre étudié ;
    - sinon le plus petit k dont la variance cumulée atteint le seuil ;
    - si le seuil n'est jamais atteint : le plus grand k étudié.
    """
    k_max = len(cumulative)
    if n_components:
        if n_components > k_max:
            raise ValueError(f"--n-components ({n_components}) dépasse k_max ({k_max})")
        return n_components
    for k, value in enumerate(cumulative, start=1):
        if value >= threshold:
            return k
    return k_max


def variance_mismatch(model_variance, curve_variance, tolerance=1e-3):
    """Vrai si la variance du modèle final et celle de la courbe diffèrent trop.

    Elles viennent de deux ajustements de la PCA : elles doivent coïncider.
    """
    return abs(model_variance - curve_variance) > tolerance


def decode_image(raw, size=IMG_SIZE):
    """Octets d'une image -> tableau float32 (hauteur, largeur, 3), en RGB, redimensionné."""
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert("RGB").resize(size), dtype=np.float32)


def stack_batch(contents, size=IMG_SIZE):
    """Lot d'images (octets) -> tableau (n, hauteur, largeur, 3), dans le même ordre."""
    return np.stack([decode_image(raw, size) for raw in contents])


def check_results(n_result, n_images, n_features, pca_length, k):
    """Vérifie les résultats écrits ; lève ValueError en listant tous les écarts."""
    problems = []
    if n_result != n_images:
        problems.append(f"{n_result} lignes écrites pour {n_images} images")
    if n_features != N_FEATURES:
        problems.append(f"{n_features} features avant la PCA au lieu de {N_FEATURES}")
    if pca_length != k:
        problems.append(f"{pca_length} valeurs après la PCA au lieu de k = {k}")
    if problems:
        raise ValueError("Validation des résultats échouée : " + " ; ".join(problems))
