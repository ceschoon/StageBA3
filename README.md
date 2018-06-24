# Stage BA3 - Hydrodynamique


Ce travail consiste d'une part à étudier les instabilités hydrodynamiques dans des écoulements de type Poiseuille, et d'autre part de s'intéresser à l'utilisation d'une carte graphique pour réaliser les calculs.

L'étude de la stabilité se divise en deux parties, modale et non-modale, qui se retrouvent dans les notebooks suivants:

- Etude modale de la stabilite.ipynb
- Etude non modale de la stabilite.ipynb

Les fichiers.py sont des modules de code utilisés dans ces notebooks.

L'utilisation de la carte graphique a été approchée avec deux librairies: magma (c) et tensorflow (python).

Une tentative d'écriture du code de l'analyse non modale en c/magma se trouve dans le dossier "svd_gpu_c/". Malheureusement, je n'ai pas pu le faire tourner sur aqua3 à cause d'une erreur de gestion de la mémoire (segmentation fault) qui survient sur cette machine et que je n'ai pas pu résoudre. J'ai pu cependant l'exécuter sur ma machine et les résultats sont discutés dans le notebook consacré à la partie non modale. La localisation de l'erreur est mise en évidence dans le fichier svd_gpu_c/svd_with_t.cpp . 

Une écriture assez rudimentaire de code avec tensorflow est traitée dans le notebook consacré à la partie non modale et se trouve dans le fichier "svd_t_tf.py".

Le notebook "Comparaisons svd.ipynb" contient une petite analyse de temps d'exécution de svd avec magma sur gpu et cpu réalisée sur ma machine. Le code source ayant servi se trouve dans le dossier "magma_tests_svd". Il est écrit pour ma machine et je n'ai pas cherché à le rendre exécutable ailleurs, celui-ci n'ayant pas une grande importance.


Réalisé par Cédric Schoonen sous la direction du professeur Bernard Knaepen.