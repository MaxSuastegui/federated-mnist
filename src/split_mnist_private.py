import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold


N_CLIENTS = 3
OUTPUT_DIR = "local_data"
RANDOM_STATE = 42


def normalize_images(x):
    """
    Normalizes MNIST images and adds the channel dimension.
    Original shape: (n_samples, 28, 28)
    Final shape: (n_samples, 28, 28, 1)
    """
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, axis=-1)
    return x


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    print("Creating 3 statistically equivalent client partitions...")

    skf = StratifiedKFold(
        n_splits=N_CLIENTS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    all_indices = np.arange(len(x_train))

    for client_id, (_, client_indices) in enumerate(skf.split(all_indices, y_train), start=1):
        x_client = x_train[client_indices]
        y_client = y_train[client_indices]

        output_path = os.path.join(OUTPUT_DIR, f"client_{client_id}.npz")

        np.savez_compressed(
            output_path,
            x_train=x_client,
            y_train=y_client,
            x_test=x_test,
            y_test=y_test,
        )

        unique, counts = np.unique(y_client, return_counts=True)
        label_distribution = dict(zip(unique.tolist(), counts.tolist()))

        print(f"\nClient {client_id}")
        print(f"  Samples: {len(x_client)}")
        print(f"  Label distribution: {label_distribution}")
        print(f"  Saved to: {output_path}")

    print("\nDone. Local client data files were created inside local_data/.")


if __name__ == "__main__":
    main()
