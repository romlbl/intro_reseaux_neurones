# Introduction aux réseaux de neurones avec NumPy

Ce dépôt regroupe des projets Python implémentés uniquement avec NumPy. Aucune bibliothèque de deep learning (PyTorch, Keras, scikit-learn, etc.) ne fait le travail à notre place : propagation avant, rétropropagation, descente de gradient, fonctions d'activation et fonctions de perte sont entièrement codées à la main.

L'objectif ici n'est pas la performance mais la compréhension de ce qui se passe.

---

## Sommaire

1. [Reconnaissance de chiffres manuscrits (MNIST)](#1--reconnaissance-de-chiffres-manuscrits)
2. [Réseau qui apprend à jouer au Puissance 4](#2--réseau-qui-apprend-à-jouer-au-puissance-4)
   
---

## 1 · Reconnaissance de chiffres manuscrits

Premier projet : un réseau de neurones entièrement connecté qui reconnaît des chiffres de 0 à 9 écrits à la main.

### Données
Le réseau est entraîné sur la base d'images **MNIST**. La bibliothèque TensorFlow est utilisée uniquement pour charger le jeu de données (jamais pour construire ou entraîner le réseau). Chaque image fait $28 \times 28$ pixels en niveaux de gris.

### Architecture

| Couche | Taille | Rôle |
| :--- | :--- | :--- |
| **Entrée** | 784 neurones ($28 \times 28$ aplatis) | Un neurone par pixel |
| **Cachée** | 64 neurones | Extraction de motifs |
| **Sortie** | 10 neurones | Un neurone par chiffre (0 à 9) |

`784 pixels` ──► `[ 64 neurones ]` ──► `[ 10 neurones ]` ──► `chiffre prédit`

### Principe de fonctionnement
1. **Propagation avant** : l'image traverse les couches (produit matriciel + biais). La fonction d'activation utilisée est ReLU.
2. **Calcul de l'erreur** : erreur quadratique moyenne entre la prédiction et la vraie valeur. 
   * *Exemple de prédiction* : `[0.08, 0.07, 0.12, 0.14, 0.12, 0.09, 0.13, 0.13, 0.06, 0.06]` (prédiction maximale à l'indice 3, soit le chiffre 3).
   * *Cible réelle* : `[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]` (le vrai chiffre est un 8).
1. **Rétropropagation (** : le gradient est calculé couche par couche, entièrement à la main.
2. **Mise à jour** : ajustement des poids et des biais par descente de gradient.

---

## 2 · Réseau qui apprend à jouer au Puissance 4

Deuxième projet, bien plus ambitieux : faire apprendre à un réseau de neurones à jouer au Puissance 4, sans aucune règle de stratégie injectée au départ.

### Modélisation
* Le plateau de Puissance 4 est modélisé par un tableau de 42 cases (7 colonnes et 6 lignes).
* Les règles et le fonctionnement du jeu sont implémentés via des fonctions Python.
* Le réseau renvoie en sortie la colonne à jouer pour le prochain coup.

### Architectures comparées
Deux architectures de réseaux sont testées et comparées :

| Couches       | Réseau A                   | Réseau B                   |
| :------------ | :------------------------- | :------------------------- |
| **Entrée**    | 42 neurones (1 par case)   | 42 neurones (1 par case)   |
| **Interne 1** | 64 neurones                | 64 neurones                |
| **Interne 2** | -                          | 32 neurones                |
| **Sortie**    | 7 neurones (1 par colonne) | 7 neurones (1 par colonne) |

* **Fonctions d'activation** : `ReLU` pour les couches cachées, `Softmax` pour la couche de sortie.

### Modes de partie pour l'entraînement
* **Réseau contre aléatoire** : l'adversaire joue des coups au hasard.
* **Réseau contre lui-même**  : le réseau s'affronte et apprend de ses propres parties.

### Méthodes d'apprentissage
Trois approches, de la plus simple à la plus fine, ont été testées :

1. **a) Apprentissage simple (Victoire / Défaite)**
   * À la fin de la partie, si le réseau gagne, tous ses coups sont valorisés (on pousse la probabilité du coup joué vers 1). S'il perd, tous ses coups sont pénalisés (vers -1).
   * Limite : Le réseau n'apprend aucune tactique fine. Tous les coups d'une partie gagnée sont récompensés, même les mauvais. Une technique simpliste comme empiler 4 pions dans la même colonne est sur-renforcée. Résultat : pas de défense, pas de stratégie globale.
1. **b) Apprentissage par récompense**
   * Chaque coup reçoit un score calculé à partir de critères tactiques (aligner des pions, bloquer l'adversaire). Ce score pondère l'impact du gradient. On évalue chaque décision individuellement.
3. **c) Apprentissage tactique (Supervisé)**
   * Pour chaque coup : on calcule le meilleur coup théorique selon le système de score, on mesure l'erreur entre le coup réellement joué et ce meilleur coup, puis on rétropropage. Le réseau apprend directement le "bon coup" à jouer.

### Problème détecté :
 En s'entraînant en *self-play*, le réseau finit systématiquement par adopter la stratégie triviale d'empiler 4 pions dans la même colonne sans contrer l'adversaire.
 
* **Cause possible** : Comme il joue contre lui-même (et qu'au début il est mauvais), l'empilement est la façon la plus rapide de trouver un schéma gagnant en 4 coups car son adversaire (lui-même ) a la même stratégie et donc ne l'empêche pas d'empiler . Il juge donc cette stratégie optimale.

* **Pistes de résolution** : Injecter davantage d'exploration aléatoire pour explorer des nouvelles stratégies et aussi car un coup aléatoire peut contrer l'empilement et donc forcer le réseau à  favoriser une autre stratégie.

###  Améliorations apportées

* **Exploration aléatoire** : pendant l'entraînement, le réseau ne choisit pas toujours aveuglément le meilleur coup estimé, il s'autorise une part d'aléa pour découvrir de nouvelles approches.

* **Personnalisation du style** : en modifiant les pondérations des scores, on peut orienter le modèle vers un style plus offensif ou défensif.
