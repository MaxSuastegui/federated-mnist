import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def load_client_data(client_id, data_dir="local_data"):
    """
    Loads the local data partition for one federated client.
    """
    path = os.path.join(data_dir, f"client_{client_id}.npz")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Client data file not found: {path}. "
            "Run src/split_mnist_private.py first."
        )

    data = np.load(path)

    return (
        data["x_train"],
        data["y_train"],
        data["x_test"],
        data["y_test"],
    )


def plot_learning_curves(history, title="Local Training Curves"):
    """
    Plots accuracy and loss curves for local training.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_model(model, x_test, y_test):
    """
    Evaluates the model and prints a classification report.
    """
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return test_loss, test_accuracy
