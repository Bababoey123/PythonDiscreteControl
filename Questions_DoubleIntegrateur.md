# Questions et exercices — Double intégrateur

---

## Notebook 1 : `DoubleIntégrateurAnalyse.ipynb`

### Configuration de la simulation

**Questions :**
- Quelle est la fréquence d'échantillonnage $f_e = 1/dt$ en Hz ?
- Combien d'instants de simulation $N = T/dt$ y aura-t-il ?

---

### Initialisation des simulateurs

**Questions :**
- Que représentent les deux composantes de l'état initial $X_0 = [[0], [0]]$ physiquement ?
- Pourquoi utilise-t-on deux simulateurs différents pour représenter le même système ?

---

### Réponse impulsionnelle discrète et continue

**Questions — après affichage :**
- Y a-t-il un écart visible entre les croix rouges (discret) et la courbe bleue (continu) ? Dans quel sens ?
- Que se passerait-il sur cet écart si l'on réduisait $dt$ ?

**Questions — analyse théorique :**
- Calculer la fonction de transfert $\frac{R(s)}{\Theta(s)}$ à partir de l'expression $\frac{d^2r(t)}{dt^2}=K\theta(t)$.
- L'allure de la réponse impulsionnelle est-elle cohérente avec la fonction de transfert ? Pourquoi ?

---

### Réponse indicielle discrète et continue

**Questions — après affichage :**
- Les points discrets (croix rouges) coïncident-ils avec la courbe continue aux instants $k \cdot dt$ ?
- Comparer visuellement avec la réponse impulsionnelle : quelle différence observe-t-on concernant l'écart entre discret et continu ?

**Questions — analyse théorique :**
- Calculer $R_{\Gamma}(s)$, la réponse de la sortie à l'échelon.
- Le résultat est-il cohérent avec la courbe affichée ci-dessus ?
- Conclure sur la stabilité du système (expliciter les pôles et les zéros de la fonction de transfert continue).

---

## Notebook 2 : `CommandeDoubleIntégrateur_PID.ipynb`

### Fonction de transfert discrète

**Questions :**
- Quel est l'ordre de la fonction de transfert discrète affichée ? Combien de pôles possède-t-elle ?
- Les pôles en boucle ouverte sont-ils à l'intérieur du cercle unité ($|z| < 1$) ? Le système est-il stable sans correcteur ?

---

### Choix des gains PID

**Questions :**
- Quel effet a une augmentation de $K_p$ sur la rapidité de la réponse ? Et sur la stabilité ?
- Quel rôle joue $K_d$ dans la réponse transitoire ?
- D'après l'explication ci-dessus, pourquoi choisit-on $K_i = 0$ ici ?

---

### Analyse en boucle fermée

**Questions :**
- Tous les rayons des pôles en boucle fermée (`Pole radii`) sont-ils inférieurs à 1 ? Le système est-il stable ?
- Le DC gain de $T_{cl}$ est-il proche de 1 ? Qu'est-ce que cela implique pour le suivi d'une consigne en échelon ?
- Un pôle réel proche de 0 est-il plus ou moins rapide qu'un pôle réel proche de 1 ?

---

### Marges de stabilité

**Questions :**
- La marge de gain affichée est-elle supérieure à la cible de 6 dB ?
- La marge de phase est-elle supérieure à la cible de 45° ?
- Que se passerait-il si l'on augmentait fortement $K_p$ ? (La marge de gain diminuerait-elle ?)

---

### Résultats de simulation

**Questions :**
- La sortie $y(t)$ atteint-elle la consigne $r = 0{,}25$ m avant $t = 3$ s ? Y a-t-il un dépassement ?
- La perturbation injectée à $t = 3$ s crée-t-elle une erreur statique permanente ? Expliquer pourquoi à l'aide de l'expression $e_\infty = D/K_p$.
- Le signal de commande $u(t)$ est-il saturé (± 10) en début de réponse ?
- D'après la courbe $F_p(z)$, vers quelle valeur la déviation due à la perturbation converge-t-elle ?

---

### Simulation non linéaire

**Questions :**
- La réponse non linéaire diffère-t-elle significativement de la réponse linéaire (temps de montée, dépassement) ?
- L'erreur statique face à la perturbation est-elle la même dans les deux simulations ?
- Quelles différences observe-t-on sur le signal de commande $u(t)$ entre les deux simulations ?

---

### Tests libres

**Test — Changer les gains du PID**

Testez plusieurs combinaisons de gains $K_p$, $K_i$, $K_d$ et observez leur influence sur la réponse de la sortie et le rejet de perturbations.

**Rappel :** remontez jusqu'à la cellule qui contient `kp`, `ki` et `kd`, modifiez leurs valeurs, puis exécutez cette cellule et toutes les cellules suivantes.

Quelle est selon vous la meilleure combinaison ? Expliquez pourquoi.

**Test — Changer $dt$**

Testez plusieurs valeurs de `dt` et observez leur influence sur la réponse de la sortie et le rejet de perturbations.

**Rappel :** remontez jusqu'à la cellule qui contient `dt`, modifiez leurs valeurs, puis exécutez cette cellule et toutes les cellules suivantes.

Que se passe-t-il ? Expliquez pourquoi.

---

## Notebook 3 : `CommandeDoubleIntégrateur_RST.ipynb`

### Fonction de transfert discrète

**Questions :**
- Quel est l'ordre de la fonction de transfert continue ? Et de la fonction discrète ?
- Les pôles discrets de la FT sont-ils à l'intérieur du cercle unité ? Qu'est-ce que cela signifie pour la stabilité en boucle ouverte ?

---

### Modèle de référence $A_m$

**Questions :**
- Le gain statique de $F_\text{des}$ est-il bien égal à 1 ? Pourquoi est-ce indispensable pour la poursuite de consigne ?
- Quelle est la valeur du rayon des pôles discrets ? Sont-ils stables ?
- Que se passe-t-il sur la rapidité de la réponse si l'on augmente $\omega_0$ ?

---

### Pôle observateur $A_0$

**Questions :**
- Le pôle de $A_0$ ($z = 0.50$) est-il plus rapide ou plus lent que les pôles dominants de $A_m$ ? Comment le vérifier ?
- Que se passerait-il si le pôle observateur était plus lent que les pôles dominants (rayon proche de 1) ?

---

### Synthèse du correcteur RST

**Questions :**
- Quel est l'ordre de $H_\text{bf}$ ? Pourquoi est-il supérieur à celui de $F_\text{des}$ ?
- Quel est l'effet de `Integrator=True` sur le polynôme $S(z)$ ? Quel facteur est ajouté ?
- Le gain statique de $H_\text{bf}$ est-il bien 1 ? Tous les rayons des pôles sont-ils inférieurs à 1 ?
- Regardez les pôles et les zéros de $H_{bf}$, lesquels vont se simplifier, à quel polynôme correspondent-ils ?

---

### Marges de stabilité

**Questions :**
- La marge de gain est-elle supérieure à 6 dB ? La marge de phase est-elle supérieure à 45° ?
- Comparer ces marges avec celles obtenues avec le correcteur PID dans le notebook précédent. Lequel est plus robuste ?

---

### Métriques de performance

**Questions :**
- Comparer le dépassement et le temps de montée obtenus ici avec ceux du correcteur PID. Lequel est plus rapide ? Lequel dépasse davantage ?
- Le dépassement observé est-il cohérent avec le coefficient d'amortissement $\zeta = 0.70$ choisi pour $A_m$ ? (utiliser l'équation de )

---

### Résultats de simulation

**Questions :**
- La sortie simulée suit-elle bien $T_\text{des}$ (courbe grise pointillée) ? Pourquoi y a-t-il un léger écart ?
- Après la perturbation à $t = 3\ \text{s}$, la sortie revient-elle exactement à la consigne ? Pourquoi ?
- La commande $u(t)$ sature-t-elle en début de réponse ? Quel impact cela a-t-il sur la réponse ?

---

### Simulation non linéaire

**Questions :**
- La réponse non linéaire diffère-t-elle de la réponse linéaire ? Dans quel sens (plus de dépassement, plus lente…) ?
- Le rejet de perturbation est-il toujours intégral (retour exact à la consigne) dans la simulation non linéaire ?
- La commande $u(t)$ présente-t-elle des différences par rapport à la simulation linéaire ?

---

### Tests de paramètres

#### 1. Fréquence propre désirée $\omega_0$

| $\omega_0$ [rad/s] | Rapidité attendue | Dépassement attendu |
|---|---|---|
| 1 | Lente | Faible |
| **2** (défaut) | Nominale | Nominal |
| 4 | Rapide | Élevé |

> Modifier `omega0` dans la cellule du modèle de référence. Que devient la marge de gain quand $\omega_0$ augmente ?

#### 2. Coefficient d'amortissement $\zeta$

| $\zeta$ | Comportement |
|---|---|
| 0.50 | Fortement sous-amorti (oscillations) |
| **0.70** (défaut) | Légèrement sous-amorti |
| 1.00 | Critique (pas de dépassement) |

> Modifier `zeta`. Observer le dépassement dans la simulation linéaire et comparer avec la valeur théorique $D\% \approx e^{-\pi\zeta/\sqrt{1-\zeta^2}} \times 100$.

#### 3. Pôle observateur $A_0$

| Pôle $A_0$ (z) | Rapidité de l'observateur |
|---|---|
| 0.95 | Trop lent (proche des pôles dominants) |
| **0.80** (défaut) | Rapide — bien séparé des pôles dominants |
| 0.30 | Très rapide (peut amplifier le bruit de mesure) |

> Modifier `A0 = np.array([1, -z_obs])`. Vérifier que tous les pôles de $H_\text{bf}$ restent bien à l'intérieur du cercle unité.

#### 4. Action intégrale (`Integrator`)

| `Integrator` | Effet |
|---|---|
| `True` (défaut) | $S(z)$ contient $(z-1)$ |
| `False` | Pas d'intégrateur |

> Passer `Integrator=False` dans l'appel à `Compute_Denominator_Matching_RST`. Observer la courbe de rejet de perturbation $F_p(z)$ : le gain statique de $F_p$ est-il encore 0 ?

#### 5. Consigne `reference`

| `reference` [m] | Observation |
|---|---|
| 0.5 | Faible déplacement, pas de saturation |
| **1.0** (défaut) | Nominal |
| 2.0 | Saturation prolongée, non-linéarités plus visibles |

> Modifier `reference` dans la cellule de simulation. Comparer les simulations linéaire et non linéaire pour une grande consigne.
