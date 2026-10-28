import numpy as np
from math import exp
from PIL import Image
from tensorflow.keras.datasets import mnist

# --- données d'entrainements ---
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X = X_train.reshape(X_train.shape[0], 28 * 28).astype("float32") / 255.0

Y = np.zeros((y_train.size, 10))
Y[np.arange(y_train.size), y_train] = 1

X = X[:20000]
Y = Y[:20000]

model = {
    #1er couche
    "W1": np.random.randn(64, 28*28) * 0.01,
    "B1": np.zeros((64, 1)),
    #2eme couche
    "W2": np.random.randn(10, 64) * 0.01,
    "B2": np.zeros((10, 1)),
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(0, x)


def derive_relu(x):
  return (x > 0).astype(float)


def couche_1(img, w, b):
  score_brute = (w @ img) + b
  activation = relu(score_brute)
  return activation, score_brute


def couche_2(activation, w, b):
  score_brute = (w @ activation) + b
  activation = sigmoid(score_brute)
  return activation


for img, label in zip(X, Y):
  img = img.reshape(-1, 1)
  label = label.reshape(-1, 1)

  # 1. Forward
  activation_1, score_brute_1 = couche_1(img, model["W1"], model["B1"])
  prediction = couche_2(activation_1, model["W2"], model["B2"])

  # 2. Erreur
  erreur = prediction - label

  # --- COUCHE 2 ---
  delta2 = erreur * prediction * (1 - prediction)
  grad_W2 = delta2 @ activation_1.T
  grad_B2 = delta2

  # --- COUCHE 1 ---
  erreur_cachee = model["W2"].T @ delta2
  delta1 = erreur_cachee * derive_relu(score_brute_1)

  # Gradients et mise à jour pour la couche 1
  grad_W1 = delta1 @ img.T
  grad_B1 = delta1

  # Mise a jour des poids et biais pour les deux couches
  model["W2"] = model["W2"] - 0.01 * grad_W2
  model["B2"] = model["B2"] - 0.01 * grad_B2
  model["W1"] = model["W1"] - 0.01 * grad_W1
  model["B1"] = model["B1"] - 0.01 * grad_B1

print("Entraînement terminé.")

img_perso = Image.open("nb_6.png").convert("L").resize((28, 28))
x_test_perso = np.array(img_perso).astype("float32")
x_test_perso = 255 - x_test_perso
x_test_perso = x_test_perso.reshape(28 * 28, 1) / 255.0

# 4. Faire la prédiction avec vos fonctions de Forward !
activation_1_test, _ = couche_1(x_test_perso, model["W1"], model["B1"])
prediction_test = couche_2(activation_1_test, model["W2"], model["B2"])

# 5. Afficher le résultat
chiffre_predit = np.argmax(prediction_test)
print(f"Mon réseau de neurones prédit que c'est le chiffre : {chiffre_predit}")
print("Scores détaillés pour chaque chiffre (de 0 à 9) :")
print(prediction_test.T)