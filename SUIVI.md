# Réunion 1: 06/05

### Prochaine réunion le 13/05

### Déroulement

#### Objectif: Parler du planning du projet, de son déroulement, et des modalités du rendu final.
Rendu toutes les semaines avec ce qu’on a fait - updates du SUIVI, réunions hebdomadaires (cf réunion mercredi prochain).

#### Approche technique:
Coder "from scratch" - choix confirmé pendant la discussion et après la réunion entre nous. Il existe des implémentations disponibles en ligne que l'on peut regarder si nécéssaire.
Utilisation de pytorch.

Projet initial: suivi du papier pour de la génération de textures.
Il existe plusieurs extensions:
* Multitexture - plusieurs types d’images.
* Textures dynamiques - coût plus élevé, à voir (trouver sources) - extension favorisée par le groupe.

Précisions: mentionner nos sources, si on reprend du code préexistant et toutes utilisations de l'IA.
Conseil d'utilisation de l'IA: surtout débuggage, commentaires… pas de génération directe à partir du papier.

Tests à faire au cours du projet:
* Tester si les opérateurs différentiels c’est crucial, pourquoi? Opérateur local? Moyenne nulle?
* Propagation autour d’un point? Sur le papier mise à jour de toutes les cellules en parallèle - ce qui constitue un gain de temps conséquent.

Loss: utiliser la référence 11 du papier (Gatys) ou IM01 pour les formules
-> peut utiliser un code préexistant, importer le VGG pré-entrainé (à faire dans un premier temps)

### Objectif minimal de la semaine
Première approche de l'importation du loss/modèle VGG. Prise en main des papiers.



# Semaine du 06/05 au 13/05

Proposition d'architecture, ajouts de fichiers de code qui définissent l'architecture du projet (non finis).
Importation dans loss.py du modèle VGG préexistant dans pytorch, voir la documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html 
NB: j'ai importé le modèle dans la fonction déstinée à cela et bloqué les paramètres vu qu'on ne va pas l'entrainer. Il est déja entrainé sur IMAGENET, qui contient pleins de textures différentes donc qui est assez approprié.
Code pour calculer les matrices de Gram et la VGG loss



# Réunion 2: 13/05

### Prochaine réunion: 20/05 

Compte rendu oral da la semaine passée.
VGG venue de l'article de Gatis: code récupéré sur son github pour ce loss. Code un peu vieux, possibles problèmes de compatibilité.
Nouvelle source pour le code de cette partie: https://storimaging.github.io/notebooksImageGeneration/ 
explication du système de couches et paires... cf papier - layer spécifiques, pas premières couches trop simples ni trop sémantiques (dernières): entre deux. 
NB: tests de loss potentiels, montrer pourquoi cette combinaison loss/architecture marche.
Code sur les matrices de Gram a identifier, tests potentiellement un peu lents.

Question: Page 4, tirage aléatoire d'un N, pourquoi cela? Diversité... - à revenir/comprendre (ourquoi on change à chaque fois)

### Objectif de la semaine: 
architecture globale et premiers tests, ainsi que prise en main de la source envoyée.



# Semaine du 13/05 au 20/05

Remplissage du code: la semaine dernière nous avions créé une architecture vide, celle-ci est maintenant complète avec une première version. La loss a aussi étée changée selon les consignes données à la dernière réunion. Une partie de débuggage a aussi étée entamée, mais nous n'avons pas trouvé toutes les sources de problèmes (ou nous n'en sommes pas sûrs). En effet certains problèmes étaient minimes ( ie "loss += w * sum(F.mse_loss(G[i], A[0]) for i in range(G.shape[0]))" n'était pas divisé par 4, ce qui changeait la learning rate en théorie), et nous ne sommes pas certains qu'il s'agisse du ou d'un problème principal du code. 
  
Si on fixe le nombre de step au minimum ca ne change pas grand chose (esthétique).
  
Fix: les couleurs semblaient dégénérées, solutionné par clamp.
  
Code des matrices de Gram recopiée et comprise.
![Texte alternatif](images/imageV1.jpeg "V1 imparfaite")



# Réunion 3: 20/05

### Prochaine réunion: à voir (mercredi potentiellement), à midi pile

Compte rendu de la semaine: tous les morceaux de code marchent ensemble. L'ensemble du papier est implémenté sauf les extensions finales. 
15/20 minutes d'execution du main (même avec GPU), jugée un peu long. A voir. Mais l'inférence est assez rapide donc étapes à séparer pour rendre cela plus rapide, qui est l'un des avantages supposé de cette méthode. Temps d'entrainement supposémment pas surprenant.
  
Matrice de Gram: regarder le reste du code.
  
Extensions potentielles: Textures dynamiques. Les sources peuvent être tirées du papier. Cela semble un peu compliqué tel quel mais il est envisageable d'imaginer une version plus simple et de s'approprier la technique. Le code est en ligne, donc vu la difficulté, le récuperer et le modifer suffit.

### Objectif de la semaine : 
Faire des tests (et comprendre les méchanismes derrière). Utiliser nos photos en plus des photos du papier. Tester les limites du modèle: filtres pas différenciels, mais tirés au hasard (attention moyenne nulle), checker ce que le papier "prétend", ou MaJ...  
  
Lire papier sur les textures dynamiques, tester leur code.
  
Il est préférable de clotûrer cette partie du code déjà faite avant de passer à autre chose (extension), donc il faut bien perfectionner, tester les limites et comprendre les propriétés importantes à garder avant de se lancer. 

