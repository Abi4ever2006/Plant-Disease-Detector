"""
Plant Leaf Disease Detection - training & evaluation
Usage:
    python train.py --data_dir data/PlantVillage --model mobilenet
    python train.py --data_dir data/PlantVillage --model cnn      (baseline)
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)

def load_data(data_dir, batch):
    common = dict(validation_split=0.3, seed=42, image_size=IMG_SIZE, batch_size=batch)
    train = tf.keras.utils.image_dataset_from_directory(data_dir, subset="training", **common)
    rest = tf.keras.utils.image_dataset_from_directory(data_dir, subset="validation", shuffle=False, **common)
    class_names = train.class_names
    n = tf.data.experimental.cardinality(rest).numpy()
    val, test = rest.take(n // 2), rest.skip(n // 2)   # 70 / 15 / 15 split
    auto = tf.data.AUTOTUNE
    return (train.prefetch(auto), val.prefetch(auto), test.prefetch(auto), class_names)

def augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ])

def build_cnn(num_classes):
    return models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        augmentation(),
        layers.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"), layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

def build_mobilenet(num_classes):
    base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = augmentation()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs), base

def plot_history(histories, out):
    acc = sum([h.history["accuracy"] for h in histories], [])
    val = sum([h.history["val_accuracy"] for h in histories], [])
    loss = sum([h.history["loss"] for h in histories], [])
    vloss = sum([h.history["val_loss"] for h in histories], [])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(acc, label="train"); ax[0].plot(val, label="val"); ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(loss, label="train"); ax[1].plot(vloss, label="val"); ax[1].set_title("Loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def evaluate(model, test_ds, class_names, out_dir):
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    # Only use labels/names for classes actually present in the test set
    labels = np.unique(y_true)
    present_names = [class_names[i] for i in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=present_names, digits=4)
    print(report)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", xticklabels=present_names, yticklabels=present_names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120); plt.close()

def resolve_data_dir(data_dir=None):
    if data_dir and os.path.exists(data_dir):
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(subdirs) >= 10:
            return data_dir
    candidates = [
        "data/PlantVillage/PlantVillage/PlantVillage",
        "data/PlantVillage/PlantVillage",
        "data/PlantVillage",
        "data",
    ]
    for c in candidates:
        if os.path.exists(c):
            subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
            if len(subdirs) >= 10:
                return c
    return data_dir or "data/PlantVillage/PlantVillage/PlantVillage"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None, help="Directory containing class subfolders")
    p.add_argument("--model", choices=["cnn", "mobilenet"], default="mobilenet")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--fine_tune_epochs", type=int, default=3)
    p.add_argument("--steps_per_epoch", type=int, default=None)
    p.add_argument("--val_steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out_dir", default="outputs")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    actual_data_dir = resolve_data_dir(a.data_dir)
    print(f"Using data directory: {actual_data_dir}")

    train, val, test, class_names = load_data(actual_data_dir, a.batch)
    print(f"{len(class_names)} classes found")
    with open(os.path.join(a.out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    model_path = os.path.join(a.out_dir, f"plant_disease_{a.model}.keras")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_best_only=False,
        monitor="val_accuracy",
        verbose=1
    )
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        checkpoint_cb
    ]
    histories = []

    fit_kwargs = {}
    if a.steps_per_epoch:
        fit_kwargs["steps_per_epoch"] = a.steps_per_epoch
    if a.val_steps:
        fit_kwargs["validation_steps"] = a.val_steps

    if a.model == "cnn":
        model = build_cnn(len(class_names))
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        histories.append(model.fit(train, validation_data=val, epochs=a.epochs * 3, callbacks=cb, **fit_kwargs))
    else:
        model, base = build_mobilenet(len(class_names))
        model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
        histories.append(model.fit(train, validation_data=val, epochs=a.epochs, callbacks=cb, **fit_kwargs))
        if a.fine_tune_epochs > 0:
            base.trainable = True                      # fine-tune top layers
            for layer in base.layers[:-30]:
                layer.trainable = False
            model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])
            histories.append(model.fit(train, validation_data=val, epochs=a.fine_tune_epochs, callbacks=cb, **fit_kwargs))

    plot_history(histories, os.path.join(a.out_dir, "training_curves.png"))
    eval_kwargs = {}
    if a.val_steps:
        eval_kwargs["steps"] = a.val_steps
    loss, acc = model.evaluate(test, **eval_kwargs)
    print(f"Test accuracy: {acc:.4f}")
    evaluate(model, test, class_names, a.out_dir)
    model.save(model_path)
    print("Saved model and reports to", a.out_dir)

if __name__ == "__main__":
    main()
