SCORE_COUP_INVALIDE = -10000.0

SCORE_VICTOIRE_IMMEDIATE = 2.0
SCORE_DOUBLE_MENACE = 1.8
SCORE_BLOCAGE_VICTOIRE = 2.0
SCORE_LAISSE_GAGNER = -2.0

SCORE_ALIGNE_TROIS = 0.8
SCORE_ALIGNE_DEUX = 0.3

SCORE_CENTRE = 0.2
SCORE_PROCHE_CENTRE = 0.1
SCORE_BORD = -0.08
SCORE_BORD_EXTREME = -0.2

SCORE_MENACE_ADVERSE_TROIS = -2
SCORE_MENACE_ADVERSE_DEUX = -1.5


def coup_possible(plateau):
    return [c for c in range(7) if plateau[c] == 0]


def jouer_coup(plateau, c, j):
    if c < 0 or c >= 7:
        return

    if plateau[c] != 0:
        return

    ind = c + 5 * 7
    while ind >= 0 and plateau[ind] != 0:
        ind -= 7

    if ind < 0:
        return

    plateau[ind] = j
    return ind

def afficher_plateau(plateau):
    symboles = {0: " ", 1: "●", -1: "○"}
    separateur = "  +" + "---+" * 7

    print("  ● Joueur 1   ○ Joueur -1")
    print("\n    " + "   ".join(str(colonne) for colonne in range(1, 8)))
    print(separateur)
    for i in range(6):
        ligne = plateau[i * 7:(i + 1) * 7]
        cases = (symboles.get(case, str(case)) for case in ligne)
        print(f"{i + 1} | " + " | ".join(f"{case:^1}" for case in cases) + " |")
        print(separateur)

def check_victoire(plateau, ind):
    j = plateau[ind]
    #les case sont 1 ou -1 selon le joueur, 0 si vide
    if j == 0:
        return False

    ligne, colonne = divmod(ind, 7)
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))

    for dl, dc in directions:
        alignement = 1
        for sens in (-1, 1):
            l, c = ligne, colonne
            while True:
                l += sens * dl
                c += sens * dc
                if not (0 <= l < 6 and 0 <= c < 7):
                    break
                if plateau[l * 7 + c] != j:
                    break
                alignement += 1
        if alignement >= 4:
            return True

    return False

def evaluer_coup(plateau, colonne, joueur):
    '''
    Évalue le score d'un coup possible selon ces tactiques :
    - 2.0 : victoire immédiate
    - 1.5 : double menace
    - 1.0 : blocage d'une victoire immédiate adverse
    - 0.6 : coupe un alignement de 3 adverse
    - 0.4 : aligne 3 pions
    - 0.2 : aligne 2 pions
    - 0.1 : joue au centre
    - 0.0 : neutre
    - -0.2 : bord / petite menace
    - -0.6 : laisse l'adversaire aligner 3 pions
    - -1.0 : laisse l'adversaire gagner immédiatement
    - -2.0 : défaite immédiate
    '''

    if colonne < 0 or colonne >= 7:
        return SCORE_COUP_INVALIDE
    if plateau[colonne] != 0:
        return SCORE_COUP_INVALIDE

    plateau_apres = plateau.copy()
    index = jouer_coup(plateau_apres, colonne, joueur)

    if check_victoire(plateau_apres, index):
        return SCORE_VICTOIRE_IMMEDIATE

    score = 0.0

    # 1) Bonus selon la colonne jouée
    if colonne == 3:
        score += SCORE_CENTRE
    elif colonne in (2, 4):
        score += SCORE_PROCHE_CENTRE
    elif colonne in (1, 5):
        score += SCORE_BORD
    else:
        score += SCORE_BORD_EXTREME

    # 2) Analyse des lignes de 4 cases autour du coup
    board = plateau_apres.reshape(6, 7)
    ligne, col = divmod(index, 7)
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))

    for dl, dc in directions:
        for sens in (-1, 1):
            cellules = []
            for k in range(1, 4):
                l = ligne + sens * dl * k
                c = col + sens * dc * k
                if 0 <= l < 6 and 0 <= c < 7:
                    cellules.append(board[l, c])
                else:
                    break
            if len(cellules) == 3:
                if cellules.count(joueur) == 3:
                    score += SCORE_ALIGNE_TROIS
                elif cellules.count(joueur) == 2 and cellules.count(0) == 1:
                    score += SCORE_ALIGNE_DEUX
                elif cellules.count(-joueur) == 3:
                    score += SCORE_MENACE_ADVERSE_TROIS
                elif cellules.count(-joueur) == 2 and cellules.count(0) == 1:
                    score += SCORE_MENACE_ADVERSE_DEUX

    # 3) Récompense le blocage d'une victoire immédiate déjà disponible.
    adversaire = -joueur
    for colonne_adv in range(7):
        if plateau[colonne_adv] != 0:
            continue

        plateau_adv = plateau.copy()
        try:
            index_adv = jouer_coup(plateau_adv, colonne_adv, adversaire)
        except ValueError:
            continue

        if check_victoire(plateau_adv, index_adv):
            score += (
                SCORE_BLOCAGE_VICTOIRE
                if colonne_adv == colonne
                else SCORE_LAISSE_GAGNER
            )

    # 4) Double menace : si ce coup crée deux possibilités de victoire immédiate
    double_menace = 0
    for colonne_test in range(7):
        if plateau_apres[colonne_test] != 0:
            continue
        plateau_tmp = plateau_apres.copy()
        try:
            index_tmp = jouer_coup(plateau_tmp, colonne_test, joueur)
        except ValueError:
            continue
        if check_victoire(plateau_tmp, index_tmp):
            double_menace += 1

    if double_menace >= 2:
        score += SCORE_DOUBLE_MENACE

    return score
