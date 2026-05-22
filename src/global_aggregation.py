import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from TheModel import build_model
from utils import load_client_data


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "local_data")
LOCAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "local_models")
GLOBAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "global_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

N_CLIENTS = 3
ROUNDS = 5
LOCAL_EPOCHS_PER_ROUND = 1
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
SEED = 42

os.makedirs(GLOBAL_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_test_data():
    """
    Loads the shared MNIST test data.

    In this simulation, every client file contains the same public test set.
    The training data is partitioned, but the test set is shared for final evaluation.
    """
    path = os.path.join(LOCAL_DATA_DIR, "client_1.npz")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Client data was not found. Run src/split_mnist_private.py first."
        )

    data = np.load(path)
    return data["x_test"], data["y_test"]


def load_client_sample_counts():
    """
    Loads the number of training samples available for each client.
    """
    path = os.path.join(RESULTS_DIR, "client_sample_counts.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "client_sample_counts.json was not found. Run src/local_training.py first."
        )

    with open(path, "r") as f:
        counts_dict = json.load(f)

    return [
        counts_dict[f"client_{client_id}"]
        for client_id in range(1, N_CLIENTS + 1)
    ]


def load_local_models():
    """
    Loads the one-shot local models previously trained in src/local_training.py.
    """
    local_models = []

    for client_id in range(1, N_CLIENTS + 1):
        model_path = os.path.join(
            LOCAL_MODELS_DIR,
            f"client_{client_id}_local.keras"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Local model not found: {model_path}. Run src/local_training.py first."
            )

        model = tf.keras.models.load_model(model_path)
        local_models.append(model)

        print(f"Loaded one-shot local model: {model_path}")

    return local_models


def normalize_weights(weights):
    """
    Normalizes a list or array of aggregation weights so that they sum to 1.
    """
    weights = np.array(weights, dtype=np.float32)
    total = np.sum(weights)

    if total == 0:
        return np.ones_like(weights) / len(weights)

    return weights / total


def aggregate_weights(local_models, aggregation_weights):
    """
    Aggregates model weights using a weighted average.
    """
    aggregation_weights = normalize_weights(aggregation_weights)
    local_weights = [model.get_weights() for model in local_models]

    aggregated_weights = []

    for layer_weights in zip(*local_weights):
        aggregated_layer = np.zeros_like(layer_weights[0])

        for client_weight, client_layer in zip(aggregation_weights, layer_weights):
            aggregated_layer += client_weight * client_layer

        aggregated_weights.append(aggregated_layer)

    global_model = build_model()
    global_model.set_weights(aggregated_weights)

    return global_model


def evaluate_model(model, x_test, y_test, method_name):
    """
    Evaluates a global model and returns summary metrics.
    """
    print("\n" + "=" * 80)
    print(f"Evaluating: {method_name}")
    print("=" * 80)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report_text = classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        digits=4,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(y_test, y_pred)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(report_text)
    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "method": method_name,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    }


def save_model(model, filename):
    """
    Saves a model inside global_models/.
    """
    path = os.path.join(GLOBAL_MODELS_DIR, filename)
    model.save(path)
    print(f"Saved model to: {path}")


def fedavg_one_shot(sample_counts):
    """
    Baseline FedAvg.

    This method uses the local models trained once by each client and aggregates them
    in a single global averaging step.
    """
    print("\n" + "=" * 80)
    print("Computing FedAvg One-Shot baseline")
    print("=" * 80)

    local_models = load_local_models()
    aggregation_weights = normalize_weights(sample_counts)

    print("Aggregation weights:", aggregation_weights.tolist())

    global_model = aggregate_weights(local_models, aggregation_weights)

    save_model(global_model, "fedavg_one_shot.keras")

    return global_model


def train_client_from_global(global_model, client_id):
    """
    Trains one client for one federated round, starting from the current global model.
    """
    x_train, y_train, _, _ = load_client_data(
        client_id=client_id,
        data_dir=LOCAL_DATA_DIR
    )

    local_model = build_model()
    local_model.set_weights(global_model.get_weights())

    history = local_model.fit(
        x_train,
        y_train,
        epochs=LOCAL_EPOCHS_PER_ROUND,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=0
    )

    final_val_accuracy = float(history.history["val_accuracy"][-1])
    final_val_loss = float(history.history["val_loss"][-1])

    return local_model, len(x_train), final_val_accuracy, final_val_loss


def federated_training_multi_round(weighting_strategy="samples"):
    """
    Runs a multi-round federated learning simulation.

    weighting_strategy:
    - "samples": weights clients by number of local samples.
    - "performance": weights clients by validation accuracy in each round.
    """
    if weighting_strategy not in {"samples", "performance"}:
        raise ValueError("weighting_strategy must be either 'samples' or 'performance'.")

    print("\n" + "=" * 80)
    print(f"Starting multi-round federated training: {weighting_strategy}")
    print("=" * 80)

    tf.keras.utils.set_random_seed(SEED)

    initial_global_path = os.path.join(GLOBAL_MODELS_DIR, "global_initial.keras")

    if os.path.exists(initial_global_path):
        global_model = tf.keras.models.load_model(initial_global_path)
        print(f"Loaded initial global model from: {initial_global_path}")
    else:
        global_model = build_model()
        global_model.save(initial_global_path)
        print(f"Created new initial global model at: {initial_global_path}")

    round_history = []

    for round_idx in range(1, ROUNDS + 1):
        print("\n" + "-" * 80)
        print(f"Federated round {round_idx}/{ROUNDS}")
        print("-" * 80)

        local_models = []
        sample_counts = []
        val_accuracies = []
        val_losses = []

        for client_id in range(1, N_CLIENTS + 1):
            local_model, samples, val_accuracy, val_loss = train_client_from_global(
                global_model,
                client_id
            )

            local_models.append(local_model)
            sample_counts.append(samples)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print(
                f"Client {client_id}: "
                f"samples={samples}, "
                f"val_accuracy={val_accuracy:.4f}, "
                f"val_loss={val_loss:.4f}"
            )

        if weighting_strategy == "samples":
            aggregation_weights = normalize_weights(sample_counts)
        else:
            aggregation_weights = normalize_weights(val_accuracies)

        print("Aggregation weights:", aggregation_weights.tolist())

        global_model = aggregate_weights(local_models, aggregation_weights)

        round_history.append(
            {
                "round": round_idx,
                "weighting_strategy": weighting_strategy,
                "client_sample_counts": [int(x) for x in sample_counts],
                "client_val_accuracies": [float(x) for x in val_accuracies],
                "client_val_losses": [float(x) for x in val_losses],
                "aggregation_weights": [float(x) for x in aggregation_weights],
            }
        )

    if weighting_strategy == "samples":
        save_model(global_model, "fedavg_multi_round.keras")
    else:
        save_model(global_model, "performance_weighted_multi_round.keras")

    return global_model, round_history


def main():
    print("=" * 80)
    print("Federated MNIST Global Aggregation")
    print("=" * 80)

    x_test, y_test = load_test_data()
    sample_counts = load_client_sample_counts()

    print("Client sample counts:", sample_counts)

    results = {}

    # Method 1: baseline FedAvg one-shot
    one_shot_model = fedavg_one_shot(sample_counts)
    results["fedavg_one_shot"] = evaluate_model(
        one_shot_model,
        x_test,
        y_test,
        "FedAvg One-Shot"
    )

    # Method 2: multi-round FedAvg using sample-count weights
    multi_round_model, multi_round_history = federated_training_multi_round(
        weighting_strategy="samples"
    )
    results["fedavg_multi_round"] = evaluate_model(
        multi_round_model,
        x_test,
        y_test,
        "FedAvg Multi-Round"
    )

    # Method 3: multi-round FedAvg using validation performance weights
    performance_model, performance_history = federated_training_multi_round(
        weighting_strategy="performance"
    )
    results["performance_weighted_multi_round"] = evaluate_model(
        performance_model,
        x_test,
        y_test,
        "Performance-Weighted Multi-Round"
    )

    final_results = {
        "configuration": {
            "n_clients": N_CLIENTS,
            "rounds": ROUNDS,
            "local_epochs_per_round": LOCAL_EPOCHS_PER_ROUND,
            "batch_size": BATCH_SIZE,
            "validation_split": VALIDATION_SPLIT,
            "seed": SEED,
        },
        "results": results,
        "round_history": {
            "fedavg_multi_round": multi_round_history,
            "performance_weighted_multi_round": performance_history,
        },
    }

    results_path = os.path.join(
        RESULTS_DIR,
        "global_aggregation_results.json"
    )

    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n" + "=" * 80)
    print("Final Summary")
    print("=" * 80)

    for key, value in results.items():
        print(
            f"{value['method']}: "
            f"accuracy={value['test_accuracy']:.4f}, "
            f"macro_f1={value['macro_f1']:.4f}, "
            f"weighted_f1={value['weighted_f1']:.4f}"
        )

    print(f"\nSaved final results to: {results_path}")


if __name__ == "__main__":
    main()
