# Details

The works of Gärdenfors [(2000)](https://doi.org/10.7551/mitpress/2076.001.0001) and [(2014)](https://doi.org/10.7551/mitpress/9629.001.0001) provide a detailed introduction to conceptual space theory.

The file [Tesselations_Voronoi.ipynb](https://github.com/ZygoOoade/conceptual_spaces/blob/main/Tesselations_Voronoi.ipynb) is used to illustrate the model proposed by [Douven (2013)](https://link.springer.com/article/10.1007/s10992-011-9216-0), and [Douven (2016)](https://doi.org/10.1016/j.cognition.2016.03.007). More detailed code is available on [this GitHub](https://github.com/IgorDouven/LearningConcepts/blob/main/learning_concepts.jl)

The file [Glove__sleep__DRM_visualisation.ipynb](https://github.com/ZygoOoade/conceptual_spaces/blob/main/Glove__sleep__DRM_visualisation.ipynb) was a draft for modeling the DRM effect. It would be better to use more recent embeddings, and also to apply MDS rather than PCA.

# TODO
## Intro
- Retirer la section sur le clustering du fichier LaTeX "nouvelle version en français" pour le mettre en annexe
- Ajouter l'ancienne introduction à la suite de la nouvelle introduction avec une transition souple


## Chapitre 1
- Traduire le passage sur les deux fonctions d'appartenance gradée de l'anglais vers le français.
 - Intégrer ce passage dans la section 1.5.2

- Décomposer en sous parties le code python "Corrélation positive entre la taille de la région prototypique et l'épaisseur de la zone de transition". Cette décomposition permettra d'y voir plus clair.
- Mettre lien vers le Colab Notebook en footnote ou bien en annexe
- Une fois cette décomposition opérée, ajouter des détails formels sur cette corrélation : la démonstration de Douven pour le cas 1-dimensionnel et mes résultats en python pour le cas 2-dimensionnel.

## Chapitre sur les DRM
- Ajouter une mention de SINr et en dire plus concernant la possibilité de démêler les dimensions


- Rédiger une partie sur la possibilité d'améliorer les relations de distance dans StyleGAN3, de manière à optimiser l'équation `D(x,y) = Dissimilarité(x,y)` et de contribuer à en faire un espace conceptuel / un similarity space.

# See also
For more details, we refer to the Python implementation of the conceptual space theory developed by Bechberger [in his thesis](https://osnadocs.ub.uni-osnabrueck.de/handle/ds-2023120110100).
