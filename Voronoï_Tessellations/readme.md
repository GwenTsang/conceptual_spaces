
Soit un ensemble générateur \( P \) composé de 22 points générateurs d'un diagramme de Voronoi. Cet ensemble est construit de la manière suivante :
   - Le premier point est placé au centre.
   - Les 21 autres points sont générés autour de ce premier point central.
   - Deux régions de Voronoi sont dites adjacentes si elles partagent au moins une arête commune.
   - Les points générant des régions qui ne partagent aucune arête avec la région du point central ne sont pas considérés comme adjacents.

Nous colorons d'une manière distincte les points dont la région est adjacente à celle du point central.
Nous générons un ensemble \( Adj\_P \), contenant uniquement les points adjacents au point central (sans inclure ce dernier).

Il existe des méthodes pour faire en sorte que la région générée par le point rouge reste inchangée malgré un changement de l'ensemble des points noirs.
Dans le code python proposé, nous employons une méthode "bricolée" qui consiste à construire l'enveloppe convexe des points bleus, et générer les points noirs en dehors.

Supposons que deux point p_i et p_r sont fixés dans un espace bidimensionnel. Soit R(p_i, p_r) la région contenant l'ensemble des points plus proches de p_i que de p_r et soit R(p_r, p_i) la région contenant l'ensemble des points plus proches de p_r que de p_i.
Une amélioration des codes python présent consisterait à trouver une méthode algorithmique pour savoir, pour tout point p_k dans le même espace bidimensionnel, si :
- R(p_k, p_i) est entièrement contenu dans R(p_r, p_i) et donc l'intersection de R(p_k, p_i) avec R(p_i, p_r) est vide.
- L'intersection de R(p_k, p_i) avec R(p_i, p_r) n'est pas vide.

Il semble intuitivement que tout point qui n'est pas sur la droite qui traverse p_i et p_r vérifie la seconde condition si l'on considère que l'espace bidimensionnel en question est infini.


   - **Définitions** : Les régions \( R(a, b) \) sont définies comme des demi-plans constitués de tous les points plus proches de \( a \N) que de \( b \N), c'est-à-dire, \( R(a, b) = \{ x \Ndans \mathbb{R}^2 : |x - a| \leq |x - b| \} \).
   - **Objectif** : Prouver que pour tout point \( p_k \N) qui n'est pas sur la ligne passant par \( p_i \N) et \( p_r \N), les régions \( R(p_k, p_i) \N et \( R(p_i, p_r) \N) ont une intersection non vide, c'est-à-dire que \( R(p_k, p_i) \Ncap R(p_i, p_r) \Nneq \Nemptyset \N).

**Placer \( p_i \) à l'Origine** :
   - Sans perte de généralité, vous supposez que \N( p_i \N) est à l'origine \N((0, 0)\N). Ceci est valable puisque toute transformation rigide qui déplace \Np_i \Nà l'origine préservera les distances entre les points, et donc la structure des bissectrices et des régions.

**Bisecteurs** :
   - Vous avez correctement identifié que les bissectrices sont essentielles. La bissectrice entre \N-( p_i \N) et \N-( p_r \N) est l'ensemble des points équidistants de \N-( p_i \N) et \N-( p_r \N), qui forme une ligne, comme indiqué :
   \[
   \sqrt{x^2 + y^2} = \sqrt{(x - x_r)^2 + (y - y_r)^2}
   \]
   Après quadrillage et simplification, on obtient l'équation de la bissectrice \( B_{i,r} \N) :
   \[
   2x x_r + 2y y_r = x_r^2 + y_r^2
   \]
   De même, on trouve l'équation de la bissectrice entre \Np_i \Net \Np_k \N, \Nb_{k,i} \Nqui est :
   \[
   2x x_k + 2y y_k = x_k^2 + y_k^2
   \]

**Intersection des demi-plans** :
   - Vous concluez à juste titre que puisque \Npour p_k \Nn'est pas sur la droite passant par \Npour p_i \Net \Np_r \N, les bissectrices \Npour B_{i,r} \Net \Npour B_{k,i} \Nne peuvent pas être parallèles - elles doivent se croiser en un seul point.
   - Par conséquent, les demi-plans \NR(p_i, p_r) \Net \NR(p_k, p_i) \Nforment un coin convexe, ce qui implique que leur intersection n'est pas vide. Cette étape est cruciale, car la convexité des régions garantit que toute intersection entre les bissectrices conduit à un chevauchement non vide des demi-plans.

**Convexité et intersection non vide** :
   - L'argument ici est correct : les régions sont convexes, et l'intersection d'ensembles convexes est également convexe, donc l'intersection doit contenir des points (à moins que les régions ne soient disjointes, ce qui, selon vous, n'arrive que si \( p_k \N) est sur la ligne passant par \( p_i \N) et \N( p_r \N)).

**Cas colinéaire** :
   - Dans le cas où \N- p_k \N- se trouve sur la ligne passant par \N- p_i \N et \N- p_r \N-, les bissectrices \N- B_{i,r} \N et \N- B_{k,i} \N peuvent coïncider ou être parallèles, et les régions peuvent ne pas se chevaucher. Cela permet de traiter correctement le cas exceptionnel où l'intersection pourrait être vide, en s'alignant sur l'hypothèse selon laquelle \Np_k \Nne doit pas être sur la ligne passant par \Np_i \Net \Np_r \Npour que l'intersection soit garantie non vide.
