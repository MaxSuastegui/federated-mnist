import os
import sys
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report

from TheModel import build_model
from utils import load_client_data


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "local_data")
LOCAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "local_models")
GLOBAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "global_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

N_CLIENTS = 3
EPOCHS = 3
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
SEED = 42

os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
os.makedirs(GLOBAL_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    tf.keras.utils.set_random_seed(SEED)

    initial_global_path = os.path.join(
        GLOBAL_MODELS_DIR,
        "global_initial.keras"
    )

    print("=" * 80)
    print("Creating shared initial global model")
    print("=" * 80)

    initial_global_model = build_model()
    initial_global_model.save(initial_global_path)

    print(f"Saved shared initial global model to: {initial_global_path}")

    local_results = []
    client_sample_counts = {}

    for client_id in range(1, N_CLIENTS + 1):
        print("\n" + "=" * 80)
        print(f"Training local model for client {client_id}")
        print("=" * 80)

        x_train, y_train, x_test, y_test = load_client_data(
            client_id=client_id,
            data_dir=LOCAL_DATA_DIR
        )

        client_sample_counts[f"client_{client_id}"] = int(len(x_train))

        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)

        # Important:
        # Every client starts from the exact same global initialization.
        model = tf.keras.models.load_model(initial_global_path)

        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=1
        )

        test_loss, test_accuracy = model.evaluate(
            x_test,
            y_test,
            verbose=0
        )

        y_pred_probs = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                y_pred,
                digits=4,
                zero_division=0
            )
        )

        model_path = os.path.join(
            LOCAL_MODELS_DIR,
            f"client_{client_id}_local.keras"
        )

        model.save(model_path)

        result = {
            "client_id": client_id,
            "samples": int(len(x_train)),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "model_path": model_path
        }

        local_results.append(result)

        print(f"Saved local model to: {model_path}")
        print(f"Client {client_id} test accuracy: {test_accuracy:.4f}")

    results_path = os.path.join(
        RESULTS_DIR,
        "local_training_results.json"
    )

    sample_counts_path = os.path.join(
        RESULTS_DIR,
        "client_sample_counts.json"
    )

    with open(results_path, "w") as f:
        json.dump(local_results, f, indent=4)

    with open(sample_counts_path, "w") as f:
        json.dump(client_sample_counts, f, indent=4)

    print("\n" + "=" * 80)
    print("Local training completed.")
    print("=" * 80)
    print(f"Saved local results to: {results_path}")
    print(f"Saved client sample counts to: {sample_counts_path}")

    print("\nSummary:")
    for result in local_results:
        print(
            f"Client {result['client_id']}: "
            f"accuracy={result['test_accuracy']:.4f}, "
            f"val_accuracy={result['final_val_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
