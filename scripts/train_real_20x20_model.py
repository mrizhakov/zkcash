import argparse
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

import tf_bionetta as tfb
from tf_bionetta.specs.backend_enums import (
    OptimizationLevel,
    ProvingBackend,
    WitnessGenerator,
)
from tf_bionetta.specs.target import TargetPlatform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a stronger 20x20 face embedding model and compile to ZK artifacts.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./datasets/faces",
        help="Directory with identity subfolders (one subfolder per person).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./compiled_circuit_v20_real",
        help="Output directory for compiled circuit artifacts.",
    )
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-threshold-pairs",
        type=int,
        default=20000,
        help="Max number of sampled pairs for threshold calibration.",
    )
    return parser.parse_args()


def build_embedding_backbone(image_size: int, embedding_dim: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(image_size, image_size, 3), name="image")
    x = tf.keras.layers.Conv2D(
        8,
        (3, 3),
        strides=(2, 2),
        padding="valid",
        activation=None,
        name="conv2d_1",
    )(inp)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.125, name="leaky_relu_1")(x)

    x = tf.keras.layers.Conv2D(
        8,
        (3, 3),
        strides=(1, 1),
        padding="valid",
        activation=None,
        name="conv2d_2",
    )(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.125, name="leaky_relu_2")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    emb = tf.keras.layers.Dense(
        embedding_dim,
        activation=None,
        name="embedding_layer",
    )(x)
    return tf.keras.Model(inp, emb, name="face_embedding_backbone")


def build_training_model(backbone: tf.keras.Model, num_classes: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=backbone.input_shape[1:], name="image")
    emb = backbone(inp)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.125, name="embedding_activation")(emb)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inp, out, name="face_classifier")


def make_datasets(
    data_dir: str,
    image_size: int,
    batch_size: int,
    val_split: float,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


def preprocess_for_train(ds: tf.data.Dataset) -> tf.data.Dataset:
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomBrightness(0.12),
            tf.keras.layers.RandomContrast(0.12),
            tf.keras.layers.RandomTranslation(height_factor=0.06, width_factor=0.06),
        ],
        name="augmentation",
    )

    def _map(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        images = aug(images, training=True)
        return images, labels

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def preprocess_for_eval(ds: tf.data.Dataset) -> tf.data.Dataset:
    return (
        ds.map(
            lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )


def gather_embeddings(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    emb_batches = []
    label_batches = []
    for images, labels in dataset:
        emb = model(images, training=False).numpy()
        emb_batches.append(emb)
        label_batches.append(labels.numpy())

    if not emb_batches:
        raise RuntimeError("No validation embeddings produced.")

    embeddings = np.concatenate(emb_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    return embeddings, labels


def calibrate_threshold(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_pairs: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n < 2:
        raise RuntimeError("Need at least 2 validation samples to calibrate threshold.")

    same_dists = []
    diff_dists = []

    sampled = 0
    for _ in range(max_pairs):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        d = embeddings[i] - embeddings[j]
        d2 = float(np.dot(d, d))
        if labels[i] == labels[j]:
            same_dists.append(d2)
        else:
            diff_dists.append(d2)
        sampled += 1

    if not same_dists or not diff_dists:
        raise RuntimeError(
            "Could not collect both same-person and different-person validation pairs."
        )

    same = np.array(same_dists, dtype=np.float64)
    diff = np.array(diff_dists, dtype=np.float64)

    # Grid search threshold on observed range.
    low = min(float(same.min()), float(diff.min()))
    high = max(float(same.max()), float(diff.max()))
    grid = np.linspace(low, high, num=2000)

    best_t = grid[0]
    best_score = -1.0
    best_far = 1.0
    best_frr = 1.0

    for t in grid:
        far = float(np.mean(diff <= t))  # false accept rate
        frr = float(np.mean(same > t))   # false reject rate
        score = 1.0 - (far + frr) / 2.0
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_far = far
            best_frr = frr

    return {
        "recommended_threshold": int(round(best_t)),
        "far": best_far,
        "frr": best_frr,
        "same_mean": float(np.mean(same)),
        "same_p95": float(np.percentile(same, 95)),
        "diff_mean": float(np.mean(diff)),
        "diff_p05": float(np.percentile(diff, 5)),
        "sampled_pairs": sampled,
    }


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    print("1) Loading real labeled face dataset...")
    train_raw, val_raw, class_names = make_datasets(
        args.data_dir,
        args.image_size,
        args.batch_size,
        args.val_split,
        args.seed,
    )
    num_classes = len(class_names)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 identities (subfolders) in dataset.")
    print(f"Found {num_classes} identities.")

    train_ds = preprocess_for_train(train_raw)
    val_ds = preprocess_for_eval(val_raw)

    print("2) Building stronger 20x20 embedding model...")
    backbone = build_embedding_backbone(args.image_size, args.embedding_dim)
    train_model = build_training_model(backbone, num_classes)

    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
    ]

    print("3) Training classifier head for discriminative embeddings...")
    train_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    print("4) Calibrating threshold on validation embedding distances...")
    emb_val, labels_val = gather_embeddings(backbone, val_ds)
    threshold_stats = calibrate_threshold(
        emb_val,
        labels_val,
        args.max_threshold_pairs,
        args.seed,
    )
    print("Recommended threshold:", threshold_stats["recommended_threshold"])
    print("FAR:", threshold_stats["far"], "FRR:", threshold_stats["frr"])

    print("5) Compiling embedding model to ZK artifacts...")
    proving_backend = ProvingBackend.GROTH16()
    bionetta_model = tfb.BionettaModel(backbone, verbose=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # test_input must be image tensor, not embedding.
    test_input = None
    for images, _ in val_ds.take(1):
        test_input = images[0].numpy()
        break

    if test_input is None:
        raise RuntimeError("Could not fetch a validation image for circuit compilation.")

    bionetta_model.compile_circuits(
        path=str(output_dir),
        test_input=test_input,
        save_weights=True,
        proving_backend=proving_backend,
        target_platform=TargetPlatform.WEB,
        witness_backend=WitnessGenerator.DEFAULT,
        optimization_level=OptimizationLevel.O1,
    )

    metadata = {
        "image_size": args.image_size,
        "embedding_dim": args.embedding_dim,
        "num_classes": num_classes,
        "class_names": class_names,
        "threshold_stats": threshold_stats,
    }
    metadata_path = output_dir / "recommended_threshold.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\n✅ Done")
    print("Compiled artifacts:", str(output_dir))
    print("Threshold metadata:", str(metadata_path))


if __name__ == "__main__":
    main()
