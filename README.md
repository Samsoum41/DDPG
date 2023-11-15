# Pendulum

Dans le notebook Keras_pendulum.ipynb, on s'intéressera à la résolution du problème du [Pendulum](https://www.gymlibrary.dev/environments/classic_control/pendulum/) disponible sur la plateforme gymnasium d'OpenAI :

<img src = "https://www.gymlibrary.dev/_images/pendulum.gif" width="520" height="520" style="margin:auto"/>

L'objectif est d'apprendre au pendule comment atteindre  la position verticale et se stabiliser. 

Initialement, nous nous sommes intéressés à l'algorithme REINFORCE. Pour cela, nous avons utilisé un modèle paramétrique présenté dans le livre 'REINFORCEMENT LEARNING' de S. Sutton et G. Barton. Dans la section 13.7, le livre présente un modèle d'apprentissage de densités gaussiennes avec des modèles d'espérance et de variance pour la politique. Nous n'avons pas réussi à obtenir un algorithme d'apprentissage efficace.

Pour résoudre ce problème, nous avons choisi d'utiliser d'autres algorithmes de renforcement par montée de gradient sur une politique paramétrique, les modèles Acteur-Critique. En particulier, l'algorithme de Deep Deterministic Gradient Policy (DDPG). 

## Algorithme DDPG

L'algorithme DDPG utilise un modèle Acteur-Critique qui permet d'apprendre sur des espaces d'action et d'état continues. La politique paramétrique est un réseau de neurone, l'acteur. Pour entrainer cette politique, nous utilisons un modèle paramétrique de la Q-value fonction, le critique. Ainsi, à chaque proposition d'action de l'acteur, le critique évalue la Q-value fonction pour l'état et l'action proposée pour déterminer si c'est un bon choix ou non. Réciproquement, le critique exploite la politique pour s'améliorer avec l'équation de Bellman, où l'action est prévue par l'acteur.

Ainsi, l'acteur et le critique s'améliorent réciproquement jusqu'à atteindre la politique et la Q-value fonction optimales.

DDPG utilise aussi 2 techniques importantes :

- L'utilisation d'un buffer stockant les anciennes étapes : On garde en mémoire tous les tuples (state, action, reward, next_state). Ensuite, on réalisera les optimisations de chacun des réseaux de neurones sur un échantillon de taille **batch_size** constante tirés uniformément dans ce buffer. On notera que le buffer n'est pas vidé entre les épisodes.
- L'utilisation d'un doublon pour chaque réseau de neurones : Si on utilise le même réseau de neurone pour  qu'il se mette à jour lui-même, il devient instable. Pour éviter cela, on utilise une copie de chaque réseau (acteur et critique) nommé target pour mettre à jour les modèles. Ensuite, les paramètres des réseaux targets sont mis à jour à chaque étape de l'algorithme par la formule $$ \theta_{target} = \tau\theta_{target} + (1 - \tau)\theta_{model} $$ où $\tau$ est proche de 1.

L'algorithme est présenté précisément dans l'article "Continuous control with deep reinforcement learning" (https://arxiv.org/abs/1509.02971). 

Voici un extrait du pseudo-code de l'algorithme proposé dans l'article : 

<img src="https://i.imgur.com/mS6iGyJ.jpg![image.png](attachment:325ff908-ecc2-4001-adc7-197bbee3e360.png)![image.png](attachment:f52d2582-7030-463f-bafb-48acf5402f01.png)![image.png](attachment:7ae732e0-9ecc-429b-a959-bfa8a80183b5.png)![image.png](attachment:b3d3957a-416b-46c9-b147-015c977b161e.png)" style="margin:auto"/>

Pour piloter l'exploration de notre agent et aussi améliorer sa stabilité, nous utiliserons du bruit pour perturber l'action de la politique à chaque fois. On peut utiliser plusieurs bruits. Ici nous en présenterons 3.

- Le bruit d'Ornstein-Uhlenbeck, solution initiale proposée avec l'algorithme DDPG dans le ARTICLE (Continuous control with deep reinforcement learning, 2015, https://arxiv.org/abs/1509.02971). En effet, le processus d'Ornstein-Uhlenbeck a pour propriété d'être gaussien à chaque instant et à générer des variables aléatoires corrélés. C'est aussi un processus de Markov. Initialement, l'algorithme DDPG ajoute une réalisation de ce processus à chaque action que le réseau de neurone action réalise.
- Le bruit gaussien, simplement généré par des variables aléatoires gaussiennes i.i.d, est une autre façon de réaliser du bruit. On ajoute une réalisation d'une gaussienne à l'action suggéré par la politique de l'acteur.


L'ensemble du code et des résultats est localisé dans le fichier Keras_pendulum.ipynb.
