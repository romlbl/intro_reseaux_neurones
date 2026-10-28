import numpy as np
import jeu
import model


def choisir_coup_ia(prediction, possibilités, epsilon=0.1):
    if np.random.random() < epsilon:
        return int(np.random.choice(possibilités))
    return int(possibilités[np.argmax(prediction[possibilités])])


def partie_autoadverse():
    plateau = np.zeros(42)
    joueur = 1
    memoire = {1: [], -1: []}
    possibilités = jeu.coup_possible(plateau)

    while possibilités:
        z1, n1, z2, proba_tampon = model.forward_pass(plateau)
        coup = choisir_coup_ia(proba_tampon, possibilités)
        memoire[joueur].append(
            (plateau.copy(),
             z1.copy(), 
             n1.copy(), 
             z2.copy(), 
             proba_tampon.copy(), 
             coup
             )
        )
        indice = jeu.jouer_coup(plateau, coup, joueur)
        if jeu.check_victoire(plateau, indice):
            return joueur, memoire
        joueur *= -1
        possibilités = jeu.coup_possible(plateau)


    return 0, memoire

def partie_random():
    plateau = np.zeros(42)
    joueur = 1
    memoire = {1: [], -1: []}
    possibilités = jeu.coup_possible(plateau)

    while possibilités:
        if joueur == -1:  # l'IA joue
            z1, n1, z2, proba_tampon = model.forward_pass(plateau)
            coup = choisir_coup_ia(proba_tampon, possibilités)
            memoire[joueur].append(
                (plateau.copy(),
                z1.copy(), 
                n1.copy(), 
                z2.copy(), 
                proba_tampon.copy(), 
                coup
                )
            )
            indice = jeu.jouer_coup(plateau, coup, joueur)
        else:  # aleatoire
            coup = np.random.choice(possibilités)
            indice = jeu.jouer_coup(plateau, coup, joueur)

        if jeu.check_victoire(plateau, indice):
            return joueur, memoire
        joueur *= -1
        possibilités = jeu.coup_possible(plateau)

    return 0, memoire

def apprentissage_simple(gagnant, memoire):
    if gagnant not in (-1, 1, 0) or gagnant == 0:
        return

    grad_W1 = np.zeros_like(model.param_model["W1"])
    grad_b1 = np.zeros_like(model.param_model["b1"])
    grad_W2 = np.zeros_like(model.param_model["W2"])
    grad_b2 = np.zeros_like(model.param_model["b2"])

    total_coups = 0

    for joueur in (-1, 1):
        historique = memoire.get(joueur, [])
        if not historique:
            continue

        recompense = 1.0 if joueur == gagnant else -1.0
        for plateau, z1, n1, _, prediction, coup in historique:
            cible = np.zeros_like(prediction)
            cible[coup] = 1
            dZ2 = recompense * (prediction - cible)

            grad_W2 += np.outer(dZ2, n1)
            grad_b2 += dZ2

            dZ1 = (model.param_model["W2"].T @ dZ2) * model.derivative_relu(z1)
            grad_W1 += np.outer(dZ1, plateau)
            grad_b1 += dZ1
            total_coups += 1

    if total_coups == 0:
        return

    taux_apprentissage = 0.01 / total_coups
    model.param_model["W1"] -= taux_apprentissage * grad_W1
    model.param_model["b1"] -= taux_apprentissage * grad_b1
    model.param_model["W2"] -= taux_apprentissage * grad_W2
    model.param_model["b2"] -= taux_apprentissage * grad_b2

def apprentissage_recompense(gagnant, memoire): 
    if gagnant not in (-1, 1, 0) or gagnant == 0:
        return

    grad_W1 = np.zeros_like(model.param_model["W1"])
    grad_b1 = np.zeros_like(model.param_model["b1"])
    grad_W2 = np.zeros_like(model.param_model["W2"])
    grad_b2 = np.zeros_like(model.param_model["b2"])

    total_coups = 0

    for joueur in (-1, 1):
        historique = memoire.get(joueur, [])
        if not historique:
            continue

        for plateau, z1, n1, _, prediction, coup in historique:
            cible = np.zeros_like(prediction)
            cible[coup] = 1
            score = jeu.evaluer_coup(plateau, coup, joueur) #score du coup joué
            dZ2 = score * (prediction - cible)

            grad_W2 += np.outer(dZ2, n1)
            grad_b2 += dZ2

            dZ1 = (model.param_model["W2"].T @ dZ2) * model.derivative_relu(z1)
            grad_W1 += np.outer(dZ1, plateau)
            grad_b1 += dZ1
            total_coups += 1

    if total_coups == 0:
        return

    taux_apprentissage = 0.01 / total_coups
    model.param_model["W1"] -= taux_apprentissage * grad_W1
    model.param_model["b1"] -= taux_apprentissage * grad_b1
    model.param_model["W2"] -= taux_apprentissage * grad_W2
    model.param_model["b2"] -= taux_apprentissage * grad_b2


def apprentissage_tactique(gagnant, memoire):
    if gagnant not in (-1, 0, 1):
        return

    grad_W1 = np.zeros_like(model.param_model["W1"])
    grad_b1 = np.zeros_like(model.param_model["b1"])
    grad_W2 = np.zeros_like(model.param_model["W2"])
    grad_b2 = np.zeros_like(model.param_model["b2"])

    total_coups = 0

    for joueur in (-1, 1):
        for plateau, z1, n1, _, prediction, _ in memoire.get(joueur, []):
            possibilités = jeu.coup_possible(plateau)
            if not possibilités:
                continue

            scores = np.array(
                [jeu.evaluer_coup(plateau, colonne, joueur)
                 for colonne in possibilités],
                dtype=float,
            )
            scores -= np.max(scores)
            poids = np.exp(scores / 0.25)

            cible = np.zeros_like(prediction)
            cible[possibilités] = poids / poids.sum()

            recompense = 1.0
            if gagnant != 0 and joueur != gagnant:
                recompense = 0.5

            dZ2 = recompense * (prediction - cible)
            grad_W2 += np.outer(dZ2, n1)
            grad_b2 += dZ2

            dZ1 = (model.param_model["W2"].T @ dZ2) * model.derivative_relu(z1)
            grad_W1 += np.outer(dZ1, plateau)
            grad_b1 += dZ1
            total_coups += 1

    if total_coups == 0:
        return

    taux_apprentissage = 0.01 / total_coups
    model.param_model["W1"] -= taux_apprentissage * grad_W1
    model.param_model["b1"] -= taux_apprentissage * grad_b1
    model.param_model["W2"] -= taux_apprentissage * grad_W2
    model.param_model["b2"] -= taux_apprentissage * grad_b2