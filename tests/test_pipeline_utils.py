"""Tests unitaires des fonctions pures de src/pipeline_utils.py.

Chaque test vérifie une seule fonction, isolée : ni Spark, ni TensorFlow, ni fichier.
Les images sont fabriquées en mémoire.
"""

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image, UnidentifiedImageError

from pipeline_utils import (
    IMG_SIZE,
    N_FEATURES,
    build_parser,
    check_results,
    decode_image,
    max_components,
    output_paths,
    select_k,
    stack_batch,
    threshold_reached,
    too_few_images,
    variance_mismatch,
    variance_table,
)


def image_bytes(mode="RGB", size=(10, 10), color=(0, 0, 0), fmt="PNG"):
    """Octets d'une image unie, fabriquée en mémoire."""
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


# --- select_k -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cumulative", "threshold", "expected"),
    [
        pytest.param([0.5, 0.8, 0.92, 0.99, 1.0], 0.90, 3, id="premier k au-dessus du seuil"),
        pytest.param([0.5, 0.9, 1.0], 0.90, 2, id="seuil atteint exactement"),
        pytest.param([0.95, 0.99, 1.0], 0.90, 1, id="seuil déjà atteint dès k = 1"),
        pytest.param([0.2, 0.4, 0.6], 0.90, 3, id="seuil jamais atteint : k maximal"),
        pytest.param([1.0], 0.90, 1, id="une seule composante"),
    ],
)
def test_select_k_from_threshold(cumulative, threshold, expected):
    assert select_k(cumulative, threshold) == expected


def test_select_k_accepts_a_pandas_series():
    assert select_k(pd.Series([0.4, 0.7, 0.95]), 0.90) == 3


def test_select_k_explicit_n_components_overrides_threshold():
    assert select_k([0.5, 0.8, 0.92, 0.99], 0.90, n_components=2) == 2


def test_select_k_explicit_n_components_can_equal_k_max():
    assert select_k([0.5, 0.8, 0.92], 0.90, n_components=3) == 3


def test_select_k_rejects_n_components_above_k_max():
    with pytest.raises(ValueError, match=r"--n-components \(5\) dépasse k_max \(3\)"):
        select_k([0.5, 0.8, 0.92], 0.90, n_components=5)


# --- threshold_reached --------------------------------------------------------------


@pytest.mark.parametrize(
    ("cumulative", "threshold", "expected"),
    [
        pytest.param([0.5, 0.95], 0.90, True, id="atteint"),
        pytest.param([0.5, 0.9], 0.90, True, id="atteint exactement"),
        pytest.param([0.5, 0.8], 0.90, False, id="jamais atteint"),
    ],
)
def test_threshold_reached(cumulative, threshold, expected):
    assert threshold_reached(cumulative, threshold) is expected


# --- max_components -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("k_max", "n_rows", "n_features", "expected"),
    [
        pytest.param(300, 22688, 1280, 300, id="beaucoup d'images : k_max demandé"),
        pytest.param(300, 100, 1280, 99, id="peu d'images : n - 1"),
        pytest.param(300, 300, 1280, 299, id="autant d'images que k_max : n - 1"),
        pytest.param(5000, 22688, 1280, 1280, id="plafonné par le nombre de features"),
    ],
)
def test_max_components(k_max, n_rows, n_features, expected):
    assert max_components(k_max, n_rows, n_features) == expected


def test_max_components_returns_an_int():
    assert isinstance(max_components(300, 22688), int)


# --- too_few_images -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_rows", "expected"),
    [
        pytest.param(300, True, id="300 images"),
        pytest.param(N_FEATURES, True, id="autant d'images que de features"),
        pytest.param(N_FEATURES + 1, False, id="une image de plus que de features"),
        pytest.param(3000, False, id="3000 images"),
    ],
)
def test_too_few_images(n_rows, expected):
    assert too_few_images(n_rows) is expected


# --- variance_table -----------------------------------------------------------------


def test_variance_table_columns_and_k_start_at_one():
    table = variance_table([0.5, 0.3, 0.2])
    assert list(table.columns) == ["k", "explained", "cumulative"]
    assert table["k"].tolist() == [1, 2, 3]


def test_variance_table_cumulative_is_running_sum():
    table = variance_table([0.5, 0.3, 0.2])
    assert table["cumulative"].tolist() == pytest.approx([0.5, 0.8, 1.0])


def test_variance_table_keeps_explained_values():
    table = variance_table([0.6, 0.4])
    assert table["explained"].tolist() == [0.6, 0.4]


def test_variance_table_cumulative_never_decreases():
    table = variance_table([0.4, 0.3, 0.2, 0.1])
    assert table["cumulative"].is_monotonic_increasing


# --- variance_mismatch --------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "curve", "expected"),
    [
        pytest.param(0.900, 0.900, False, id="identiques"),
        pytest.param(0.9005, 0.9000, False, id="écart sous la tolérance"),
        pytest.param(0.820, 0.901, True, id="écart d'une PCA peu fiable"),
        pytest.param(0.901, 0.820, True, id="écart dans l'autre sens"),
    ],
)
def test_variance_mismatch(model, curve, expected):
    assert variance_mismatch(model, curve) is expected


def test_variance_mismatch_custom_tolerance():
    assert variance_mismatch(0.90, 0.85, tolerance=0.1) is False


# --- output_paths -------------------------------------------------------------------


def test_output_paths_local_folder():
    assert output_paths("output") == {
        "features": "output/Results",
        "pca": "output/Results_PCA",
        "variance": "output/Results_PCA_variance",
    }


def test_output_paths_object_storage_url():
    assert output_paths("s3://bucket")["pca"] == "s3://bucket/Results_PCA"


# --- decode_image -------------------------------------------------------------------


def test_decode_image_shape_dtype_and_range():
    array = decode_image(image_bytes(size=(100, 100), color=(10, 200, 30), fmt="JPEG"))
    assert array.shape == (*IMG_SIZE, 3)
    assert array.dtype == np.float32
    assert array.min() >= 0
    assert array.max() <= 255


def test_decode_image_keeps_the_colour_of_a_solid_image():
    array = decode_image(image_bytes(color=(255, 0, 0)))
    assert np.allclose(array, [255, 0, 0], atol=1)


def test_decode_image_converts_greyscale_to_three_equal_channels():
    array = decode_image(image_bytes(mode="L", color=128))
    assert array.shape[-1] == 3
    assert np.allclose(array[..., 0], array[..., 1])
    assert np.allclose(array[..., 1], array[..., 2])
    assert np.allclose(array, 128, atol=1)


def test_decode_image_drops_the_alpha_channel():
    array = decode_image(image_bytes(mode="RGBA", color=(10, 20, 30, 255)))
    assert array.shape[-1] == 3
    assert np.allclose(array, [10, 20, 30], atol=1)


def test_decode_image_size_is_width_then_height():
    array = decode_image(image_bytes(), size=(8, 6))
    assert array.shape == (6, 8, 3)


def test_decode_image_rejects_bytes_that_are_not_an_image():
    with pytest.raises(UnidentifiedImageError):
        decode_image(b"ceci n'est pas une image")


# --- stack_batch --------------------------------------------------------------------


def test_stack_batch_shape():
    batch = stack_batch([image_bytes(), image_bytes(), image_bytes()], size=(4, 4))
    assert batch.shape == (3, 4, 4, 3)


def test_stack_batch_keeps_the_order_of_the_images():
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    batch = stack_batch([image_bytes(color=c) for c in colours], size=(4, 4))
    for index, colour in enumerate(colours):
        assert np.allclose(batch[index, 0, 0], colour, atol=1)


def test_stack_batch_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        stack_batch([])


# --- check_results ------------------------------------------------------------------


def test_check_results_accepts_consistent_results():
    check_results(n_result=3000, n_images=3000, n_features=N_FEATURES, pca_length=170, k=170)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"n_result": 2999, "n_images": 3000, "n_features": N_FEATURES, "pca_length": 5, "k": 5},
            "2999 lignes écrites pour 3000 images",
            id="lignes perdues",
        ),
        pytest.param(
            {"n_result": 10, "n_images": 10, "n_features": 512, "pca_length": 5, "k": 5},
            "512 features avant la PCA",
            id="mauvaise taille avant la PCA",
        ),
        pytest.param(
            {"n_result": 10, "n_images": 10, "n_features": N_FEATURES, "pca_length": 4, "k": 5},
            "4 valeurs après la PCA au lieu de k = 5",
            id="mauvaise taille après la PCA",
        ),
    ],
)
def test_check_results_reports_each_problem(kwargs, message):
    with pytest.raises(ValueError, match=message):
        check_results(**kwargs)


def test_check_results_lists_all_problems_at_once():
    with pytest.raises(ValueError) as error:
        check_results(n_result=1, n_images=2, n_features=3, pca_length=4, k=5)
    text = str(error.value)
    assert "1 lignes écrites pour 2 images" in text
    assert "3 features avant la PCA" in text
    assert "4 valeurs après la PCA" in text


# --- build_parser -------------------------------------------------------------------


def parse(*arguments):
    return build_parser().parse_args(["--input", "in", "--output", "out", *arguments])


def test_parser_defaults():
    args = parse()
    assert args.input == "in"
    assert args.output == "out"
    assert args.local is False
    assert args.cores == 4
    assert args.limit == 0
    assert args.threshold == 0.90
    assert args.n_components is None
    assert args.k_max == 300
    assert args.partitions is None


def test_parser_overrides_and_types():
    args = parse(
        "--local", "--cores", "2", "--limit", "3000", "--threshold", "0.95", "--n-components", "50"
    )
    assert args.local is True
    assert args.cores == 2
    assert args.limit == 3000
    assert args.threshold == 0.95
    assert args.n_components == 50


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param([], id="sans arguments"),
        pytest.param(["--input", "in"], id="sans --output"),
        pytest.param(["--output", "out"], id="sans --input"),
        pytest.param(
            ["--input", "in", "--output", "out", "--cores", "beaucoup"], id="cores non numérique"
        ),
        pytest.param(["--input", "in", "--output", "out", "--inconnu"], id="option inconnue"),
    ],
)
def test_parser_rejects_invalid_command_lines(arguments):
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)
