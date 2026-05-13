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
