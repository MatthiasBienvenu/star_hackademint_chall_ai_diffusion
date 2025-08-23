Challenge 10fusions pour le CTF Star Hackademint 2025

Ce challenge est à propos des modèles de diffusion stable inconditionnels (sans prompt).
Le but est d'utiliser un modèle présenté comme un simple débruiteur et de l'utiliser
pour générer des badges. Une fois tous ces badges générés, les seuls badges au fond rouge
sont identiques et correspondent au flag.

Pour mettre cela en place, il faut entrainer une IA de diffusion stable à générer des
badges de ce format. Il faut donc créer un dataset de badges sur lequel on va entrainer
un U-Net assez classique.
