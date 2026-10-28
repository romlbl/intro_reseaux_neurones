import apprentissage as app
import apprentissage2 as app2
import jeu
import model
import numpy as np
import random as rd

#apprentissage des parties
iterations = 70000
for i in range(iterations):
    if rd.random() < 0.1:
        gagnant, memoire  = app2.partie_autoadverse()
    else:
        gagnant, memoire  = app2.partie_random()
    app2.apprentissage_tactique(gagnant, memoire)

print("Entraînement terminé.")


#jeu fasse a l'IA
def jouer_contre_ia():
    plateau = np.zeros(42)
    joueur = 1
    possibilités = jeu.coup_possible(plateau)
    while possibilités != []:
        if joueur == -1: #l'IA joue
            z1, n1, z2, proba_tampon = model.forward_pass(plateau)
            coup = possibilités[np.argmax(proba_tampon[possibilités])]
        else: #le joueur humain joue
            coup = int(input("Numéro colonne (1-7) : ")) - 1

        indice = jeu.jouer_coup(plateau, coup, joueur)
        jeu.afficher_plateau(plateau)
        print("----------")
        if jeu.check_victoire(plateau, indice):
            print(f"Le joueur {joueur} a gagné !")
            break
        joueur *= -1
        possibilités = jeu.coup_possible(plateau)

def jouer_contre_ia_mod2():
    plateau = np.zeros(42)
    joueur = 1
    possibilités = jeu.coup_possible(plateau)
    while possibilités != []:
        if joueur == -1: #l'IA joue
            z1, n1, z2, n2, z3, proba_tampon = model.forward_pass_2(plateau)
            coup = possibilités[np.argmax(proba_tampon[possibilités])]
        else: #le joueur humain joue
            coup = int(input("Numéro colonne (1-7) : ")) - 1

        indice = jeu.jouer_coup(plateau, coup, joueur)
        jeu.afficher_plateau(plateau)
        print("----------")
        if jeu.check_victoire(plateau, indice):
            print(f"Le joueur {joueur} a gagné !")
            break
        joueur *= -1
        possibilités = jeu.coup_possible(plateau)

while True:
    rejouer = input("Lancer partie (o/n) : ")
    if rejouer.lower() == "o":
        jouer_contre_ia()
    else:
        break
