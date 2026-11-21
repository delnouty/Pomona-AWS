"""Chaîne de traitement PySpark du projet Fruits!

Étapes :
    1. Lecture des images (binaryFile) et extraction du label depuis le dossier parent
    2. Extraction de features avec MobileNetV2 (sans la tête de classification)
    3. Centrage des features, puis PCA (pyspark.ml) ; k = plus petit nombre de
       composantes qui conserve 90 % de la variance
    4. Écriture des résultats : Results (features), Results_PCA (features réduites)
       et Results_PCA_variance (courbe de variance expliquée, CSV)
    5. Validation : mêmes nombres de lignes, 1280 valeurs avant la PCA, k après

Le même script tourne en local (chemins locaux) et sur EMR (chemins s3://).

Exemples :
    python src/pipeline.py --local --input data/Test --output output --limit 300
    spark-submit src/pipeline.py --input s3://<bucket>/Test --output s3://<bucket>
"""

import time
from pathlib import Path
from typing import Iterator

import pandas as pd
from pyspark.ml.feature import PCA, StandardScaler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf

from pipeline_utils import (
    IMG_SIZE,
    N_FEATURES,
    build_parser,
    check_results,
    max_components,
    output_paths,
    select_k,
    stack_batch,
    threshold_reached,
    too_few_images,
    variance_mismatch,
    variance_table,
)


def build_model(weights=None):
    """MobileNetV2 sans tête de classification ; sortie = vecteur de 1280 valeurs.

    On construit toujours le modèle des executors avec weights=None : les poids
    ImageNet sont injectés ensuite via set_weights, à partir de la variable
    broadcastée. Ainsi aucun worker n'a besoin de télécharger quoi que ce soit.
    """
    # import ici : TensorFlow n'est chargé que là où il est utilisé (driver / workers)
    from keras.applications import MobileNetV2

    return MobileNetV2(
        weights=weights, include_top=False, pooling="avg", input_shape=IMG_SIZE + (3,)
    )


def make_featurize_udf(broadcast_weights):
    """Crée la pandas UDF d'extraction de features.

    La version "Iterator" charge le modèle une seule fois par partition, et non
    une fois par lot d'images.
    """

    @pandas_udf("array<float>")
    def featurize(content_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        from keras.applications import mobilenet_v2

        preprocess_input = mobilenet_v2.preprocess_input

        model = build_model()
        model.set_weights(broadcast_weights.value)

        for contents in content_iter:
            features = model.predict(preprocess_input(stack_batch(contents)), verbose=0)
            yield pd.Series(list(features))

    return featurize


def build_spark(local, cores):
    builder = (
        SparkSession.builder.appName("fruits-pipeline")
        # 256 images par lot Arrow : ~150 Mo une fois décodées (le défaut,
        # 10 000, en demanderait ~6 Go côté Python)
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "256")
    )
    if local:
        # Sur EMR ces réglages viennent du cluster : ne pas les forcer.
        # Chaque tâche lance un worker Python qui charge TensorFlow (~1 Go) : local[*]
        # sur une machine à 22 cores sature la mémoire, d'où un nombre de cores borné.
        arrow_java = "-Dio.netty.tryReflectionSetAccessible=true"  # Arrow sous Java 17
        builder = (
            builder.master(f"local[{cores}]")
            .config("spark.driver.memory", "4g")
            .config("spark.driver.extraJavaOptions", arrow_java)
            .config("spark.executor.extraJavaOptions", arrow_java)
        )
    return builder.getOrCreate()


def main():
    args = build_parser(__doc__).parse_args()
    paths = output_paths(args.output)

    spark = build_spark(args.local, args.cores)
    sc = spark.sparkContext
    # Les workers Python importent pipeline_utils (utilisé par l'UDF) : on leur envoie le module
    sc.addPyFile(str(Path(__file__).with_name("pipeline_utils.py")))

    # 1. Images + labels (le label est le nom du dossier parent de chaque image)
    images = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(args.input)
        .withColumn("label", F.element_at(F.split("path", "/"), -2))
    )
    if args.limit:
        # tirage aléatoire (graine fixe) plutôt que les N premiers fichiers, qui
        # viendraient des premières classes seulement
        images = images.orderBy(F.rand(42)).limit(args.limit)
    n_images = images.count()
    n_labels = images.select("label").distinct().count()
    print(f"{n_images} images, {n_labels} classes, {sc.defaultParallelism} cores")

    # 2. Broadcast des poids : envoyés une fois à chaque executor au lieu d'être
    # re-sérialisés avec chaque tâche
    weights = build_model(weights="imagenet").get_weights()
    broadcast_weights = sc.broadcast(weights)
    print(
        len(weights),
        "tableaux de poids,",
        round(sum(w.nbytes for w in weights) / 1e6, 1),
        "Mo diffusés",
    )
    featurize = make_featurize_udf(broadcast_weights)

    # binaryFile découpe par taille de fichier : ~22k petites images tiennent dans
    # 1 ou 2 partitions, donc sans repartition un seul core travaillerait. 2 par core
    # s'adapte à la taille du cluster, contrairement à un nombre fixe
    n_partitions = args.partitions or 2 * sc.defaultParallelism
    features_df = images.repartition(n_partitions).select(
        "path", "label", featurize("content").alias("features")
    )

    t0 = time.time()
    features_df.write.mode("overwrite").parquet(paths["features"])
    seconds_features = round(time.time() - t0, 1)
    print(f"Extraction des features : {seconds_features} s")

    # 3. On relit le parquet (point de reprise : évite de recalculer le CNN), puis
    # on centre. Le fit de la PCA de Spark est correct (matrice de covariance), mais
    # son transform ne soustrait pas la moyenne : sans centrage, les coordonnées
    # enregistrées seraient décalées d'un vecteur constant.
    t0 = time.time()
    features = spark.read.parquet(paths["features"]).withColumn(
        "features_vec", array_to_vector("features")
    )
    scaler = StandardScaler(
        inputCol="features_vec", outputCol="features_centered", withMean=True, withStd=False
    ).fit(features)
    centered = scaler.transform(features).cache()  # réutilisé par plusieurs actions
    n_rows = centered.count()
    if too_few_images(n_rows):
        # Mesuré : avec moins d'images que de features la matrice de covariance est de
        # rang n-1 et la PCA de Spark peut renvoyer des composantes fausses, qui
        # changent d'un lancement à l'autre. Au-delà de 1280 images, elle est exacte.
        print(
            f"ATTENTION : {n_rows} images <= {N_FEATURES} features, résultats de la PCA "
            f"peu fiables ; utiliser au moins ~1500 images (--limit 3000)"
        )

    # Courbe de variance : au plus n-1 composantes portent de la variance
    k_max = max_components(args.k_max, n_rows)
    pca_full = PCA(k=k_max, inputCol="features_centered", outputCol="pca_tmp").fit(centered)
    variance = variance_table(pca_full.explainedVariance.toArray())
    (
        spark.createDataFrame(variance)
        .coalesce(1)
        .write.mode("overwrite")
        .csv(paths["variance"], header=True)
    )

    try:
        k = select_k(variance.cumulative, args.threshold, args.n_components)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not args.n_components and not threshold_reached(variance.cumulative, args.threshold):
        print(f"Seuil de {args.threshold:.0%} non atteint avec {k_max} composantes")
    curve_variance = float(variance.cumulative.iloc[k - 1])
    print(f"k = {k} composantes conservent {curve_variance:.1%} de la variance")

    pca_model = PCA(k=k, inputCol="features_centered", outputCol="pca_vec").fit(centered)
    pca_df = (
        pca_model.transform(centered)
        .withColumn("pca_features", vector_to_array("pca_vec"))
        .select("path", "label", "pca_features")
    )
    pca_df.write.mode("overwrite").parquet(paths["pca"])
    centered.unpersist()
    seconds_pca = round(time.time() - t0, 1)
    variance_kept = float(pca_model.explainedVariance.toArray().sum())
    print(f"PCA écrite en {seconds_pca} s ; variance conservée : {variance_kept:.1%}")
    # Le modèle final et la courbe viennent de deux ajustements : ils doivent coïncider
    if variance_mismatch(variance_kept, curve_variance):
        print(
            f"ATTENTION : variance du modèle ({variance_kept:.1%}) différente de la "
            f"courbe ({curve_variance:.1%}) : PCA non fiable, "
            f"ne pas utiliser ces résultats"
        )

    # 4. Validation : on relit ce qui a été écrit, sans tout ramener sur le driver
    result = spark.read.parquet(paths["pca"])
    n_result = result.count()
    sample = result.limit(5).toPandas()
    n_features = len(spark.read.parquet(paths["features"]).first()["features"])
    pca_length = len(sample.loc[0, "pca_features"])
    print(f"lignes : {n_result} | attendu : {n_images}")
    print(f"features avant PCA : {n_features} | après PCA : {pca_length} | k = {k}")
    check_results(n_result, n_images, n_features, pca_length, k)
    print(f"images : {n_images} | classes : {n_labels} | cores : {sc.defaultParallelism}")
    print(
        f"extraction : {seconds_features} s | PCA : {seconds_pca} s | k = {k} "
        f"| variance conservée : {variance_kept:.1%}"
    )

    spark.stop()


if __name__ == "__main__":
    main()
