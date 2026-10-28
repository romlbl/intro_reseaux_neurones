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
        z1, n1, z2, n2, z3, proba_tampon = model.forward_pass_2(plateau)
        coup = choisir_coup_ia(proba_tampon, possibilités)
        memoire[joueur].append(
            (plateau.copy(),
             z1.copy(), 
             n1.copy(), 
             z2.copy(), 
             n2.copy(),
             z3.copy(),
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
            z1, n1, z2, n2, z3, proba_tampon = model.forward_pass_2(plateau)
            coup = choisir_coup_ia(proba_tampon, possibilités)
            memoire[joueur].append(
                (plateau.copy(),
                z1.copy(), 
                n1.copy(), 
                z2.copy(), 
                n2.copy(),
                z3.copy(),
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



def apprentissage_tactique(gagnant, memoire):
    if gagnant not in (-1, 0, 1):
        return

    grad_W1 = np.zeros_like(model.param_model_2["W1"])
    grad_b1 = np.zeros_like(model.param_model_2["b1"])
    grad_W2 = np.zeros_like(model.param_model_2["W2"])
    grad_b2 = np.zeros_like(model.param_model_2["b2"])
    grad_W3 = np.zeros_like(model.param_model_2["W3"])
    grad_b3 = np.zeros_like(model.param_model_2["b3"])

    total_coups = 0

    for joueur in (-1, 1):
        for plateau, z1, n1, z2, n2, _, prediction, _  in memoire.get(joueur, []):
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

            #couche 3
            dZ3 = recompense * (prediction - cible)
            grad_W3 += np.outer(dZ3, n2)
            grad_b3 += dZ3

            #couche 2
            dZ2 = (model.param_model_2["W3"].T @ dZ3) * model.derivative_relu(z2)
            grad_W2 += np.outer(dZ2, n1)
            grad_b2 += dZ2

            #couche 3
            dZ1 = (model.param_model_2["W2"].T @ dZ2) * model.derivative_relu(z1)
            grad_W1 += np.outer(dZ1, plateau)
            grad_b1 += dZ1

            total_coups += 1

    if total_coups == 0:
        return

    taux_apprentissage = 0.01 / total_coups
    model.param_model_2["W1"] -= taux_apprentissage * grad_W1
    model.param_model_2["b1"] -= taux_apprentissage * grad_b1
    model.param_model_2["W2"] -= taux_apprentissage * grad_W2
    model.param_model_2["b2"] -= taux_apprentissage * grad_b2
    model.param_model_2["W3"] -= taux_apprentissage * grad_W3
    model.param_model_2["b3"] -= taux_apprentissage * grad_b3