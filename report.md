# Rapport technique du POC de fine-tuning medical bilingue

## 1. Introduction

Ce document presente une synthese technique et critique du projet actuel de fine-tuning medical bilingue francais/anglais autour de `Qwen3-1.7B`. Le POC couvre quatre volets principaux :

- la preparation d'un dataset SFT et d'un dataset DPO a partir de sources medicales publiques ;
- l'entrainement d'un adaptateur LoRA en 4-bit sur la base `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` ;
- une evaluation comparative entre le modele de base et un checkpoint SFT ;
- une trajectoire de deploiement via une API FastAPI connectee a un serveur vLLM.

Ce rapport ne se limite pas a presenter quelques metriques. Il cherche aussi a expliquer les choix methodologiques, a situer le niveau de maturite reel du systeme, a documenter les aspects cout/performance et a exposer clairement les limites actuelles du POC.

## 2. Perimetre et etat actuel du projet

Le depot correspond a un POC pedagogique deja bien avance sur les volets donnees, fine-tuning et evaluation, mais encore en construction sur la partie benchmarking d'inference et sur l'industrialisation du deploiement.

Les elements actuellement disponibles dans le projet sont les suivants :

- un pipeline notebook pour l'EDA, la normalisation, la deduplication et l'anonymisation des donnees ;
- un notebook d'entrainement SFT puis DPO avec Unsloth, PEFT et TRL ;
- un notebook d'evaluation comparative sur `500` exemples ;
- une API FastAPI minimale avec backend `echo` pour les tests locaux et backend `vllm` pour l'inference cible ;
- un script `scripts/benchmark.py` prevu pour mesurer la latence et le debit en inference ;
- une base de documentation sur le deploiement Google Cloud et l'usage de vLLM.

En revanche, les mesures consolidees de latence, de debit, de consommation GPU et les journaux complets de duree d'entrainement ne sont pas archives dans le workspace actuel. Il faut donc garder une distinction simple en tete : certaines conclusions s'appuient sur des artefacts d'evaluation bien presents, tandis que la partie cout/performance d'infrastructure reste pour l'instant davantage au niveau du cadrage et de la methode qu'au niveau d'un benchmark completement finalise.

## 3. Methodologie

### 3.1 Preparation des donnees

Le projet assemble des sources medicales heterogenes en francais et en anglais afin de couvrir a la fois des questions ouvertes et des QCM. Les sources mentionnees dans le depot sont notamment `ANR-MALADES/MediQAl`, `nthngdy/frenchmedmcqa`, `keivalya/MedQuad-MedicalQnADataset` et `TsinghuaC3I/UltraMedical-Preference`.

Pour un POC, le pipeline de preparation suit une logique claire et raisonnable :

1. chargement et inspection des jeux de donnees ;
2. normalisation vers un schema commun ;
3. construction des exemples SFT et DPO ;
4. deduplication des couples texte/reponse ;
5. echantillonnage par quotas ;
6. separation `train/validation/test` pour le SFT ;
7. anonymisation heuristique de certaines entites sensibles.

Cette approche presente trois avantages concrets. D'abord, elle permet de melanger des formats de supervision differents tout en conservant une interface commune. Ensuite, elle rend possible une evaluation distincte entre generation libre et QCM. Enfin, elle introduit un premier niveau de reduction du risque PII, meme si cette anonymisation reste encore insuffisante pour un contexte clinique reel.

### 3.2 Choix du modele de base

Le choix de `Qwen3-1.7B` ne vient pas d'une comparaison ouverte entre plusieurs modeles candidats : il etait impose par le cadre du projet. La vraie question technique est donc plus simple : est-ce un point de depart coherent pour un POC frugal, et reste-t-il defendable dans l'optique d'un passage a l'echelle sur une infrastructure raisonnable ?

Dans ce cadre contraint, les principaux arguments en faveur de ce socle restent les suivants :

- la taille `1.7B`, suffisamment compacte pour rendre le fine-tuning et l'inference envisageables sur une infrastructure GPU modeste ;
- la disponibilite d'une variante `unsloth-bnb-4bit`, qui reduit fortement la pression memoire ;
- un bon alignement avec un usage bilingue instructionnel ;
- un cout de serving bien plus faible que celui d'un modele beaucoup plus grand, a performance potentiellement moins rentable dans le cadre du POC.

En pratique, meme si le choix initial etait impose, il reste compatible avec une logique d'iterabilite experimentale, de reproductibilite et de budget maitrise. Pour un POC qui cherche ensuite a monter en charge de facon pragmatique, ce n'est peut-etre pas le meilleur modele possible, mais c'est un point de depart tout a fait defensable.

### 3.3 Choix des techniques d'entrainement

Le projet combine deux techniques complementaires :

- `LoRA` pour le fine-tuning supervise ;
- `DPO` pour l'alignement par preferences.

Le recours a LoRA est bien adapte a ce contexte, a la fois pour des raisons de cout et de simplicite. En evitant de mettre a jour tous les poids du modele, LoRA reduit fortement la consommation memoire et le temps d'entrainement, tout en permettant une specialisation efficace sur le domaine medical. Le chargement en 4-bit pousse encore plus loin cette logique de frugalite.

Le recours a DPO se defend egalement bien. Une fois un premier alignement supervise obtenu, DPO permet de travailler la preference entre une bonne et une moins bonne reponse sans passer par un pipeline RLHF complet, beaucoup plus lourd a mettre en oeuvre. Pour un POC, c'est une facon assez naturelle d'explorer un second niveau d'alignement clinique tout en gardant une complexite operationnelle maitrisee.

### 3.4 Choix du deploiement et de vLLM

Le projet retient une architecture a deux niveaux :

- une API FastAPI qui porte le contrat applicatif ;
- un serveur vLLM pour l'inference du modele.

Ce choix tient bien la route sur le plan de la performance comme sur celui de l'architecture. vLLM est bien adapte au serving LLM : il apporte un serveur OpenAI-compatible, un meilleur debit que des appels de generation plus naifs, une gestion plus efficace du KV cache via PagedAttention et un batching continu utile en cas de concurrence. FastAPI reste de son cote une bonne couche d'integration pour porter le contrat HTTP, les routes `/health` et `/generate`, les logs et d'eventuelles regles metier.

Le backend `echo` conserve dans le depot ne doit donc pas etre interprete comme un backend de production. Il sert surtout de fallback tres leger pour la CI, les tests et le packaging Docker sans GPU.

## 4. Configuration experimentale

### 4.1 Jeux de donnees cibles

Le projet decrit deux datasets finaux :

- `Maphe/medical-sft-5k` pour l'entrainement supervise ;
- `Maphe/medical-dpo-5k` pour l'alignement par preferences.

Pour le SFT, la cible documentee est de `5 000` exemples d'entrainement, `500` exemples de validation et `500` exemples de test. L'evaluation disponible porte sur le split de test de `500` exemples, avec deux sous-ensembles :

- `267` exemples de texte libre ;
- `233` exemples de QCM.

### 4.2 Hyperparametres documentes

Le depot documente notamment les hyperparametres suivants.

Pour LoRA :

- `r = 16`
- `lora_alpha = 16`
- `lora_dropout = 0`
- `bias = none`
- modules cibles sur les projections d'attention et de MLP

Pour le SFT :

- `2` epochs
- batch size par device `32`
- gradient accumulation `16`
- learning rate `2e-4`
- optimiseur `adamw_8bit`
- sequence length `1024`

Pour le DPO :

- `1` epoch
- batch size par device `4`
- gradient accumulation `8`
- learning rate `5e-5`
- `beta = 0.1`
- optimiseur `adamw_8bit`
- sequence length `1024`

Dans l'ensemble, cette configuration est coherente avec un objectif de fine-tuning efficace sur materiel limite. Elle favorise l'experimentation rapide, meme si elle ne garantit pas a elle seule la stabilite ni la meilleure qualite clinique possible.

## 5. Resultats observes

### 5.1 Resultats quantitatifs disponibles

L'artefact de comparaison le plus complet actuellement visible dans le workspace est le resume `notebooks/eval_results/qwen3_base_vs_sft_output_summary.json`. Il compare le modele de base au checkpoint SFT `checkpoint-625` sur `500` exemples et integre, pour le texte libre, `METEOR`, la similarite cosinus et la distance euclidienne.

Les principaux chiffres sont les suivants :

| Metrique | Modele de base | Modele fine-tune SFT | Delta |
|---|---:|---:|---:|
| METEOR moyen texte libre | 0.1361 | 0.1653 | +0.0292 |
| Similarite cosinus moyenne texte libre | 0.3762 | 0.3859 | +0.0097 |
| Distance euclidienne moyenne texte libre | 0.2278 | 0.2231 | -0.0047 |
| Score QCM first-letter | 0.0515 | 0.4378 | +0.3863 |
| QCM corrects | 12 | 102 | +90 |

Pour la distance euclidienne, une baisse est une amelioration.

Ce resume fait aussi ressortir quelques tendances utiles :

- en texte libre, `METEOR` progresse nettement, la similarite cosinus n'augmente que legerement en moyenne et la distance euclidienne baisse legerement ;
- l'analyse par exemple reste toutefois partagee : sur `267` cas de texte libre, le fine-tune gagne `139` cas sur `METEOR`, seulement `113` sur la similarite cosinus contre `141` pour le modele de base, mais `159` sur la distance euclidienne contre `108` pour le modele de base ;
- en QCM, le gain est tres net et constitue le signal positif le plus robuste du POC ;
- globalement, le rang agrege reste favorable au checkpoint SFT, credite de `368` rangs `1` contre `269` pour le modele de base, avec `137` egalites.

### 5.2 Interpretation des resultats

L'amelioration la plus convaincante concerne les QCM. Le passage de `12/233` a `102/233` bonnes reponses montre que le fine-tuning a nettement renforce la capacite du modele a suivre un format de reponse contraint et a mieux selectionner l'information attendue. C'est un resultat utile, car une grande partie des jeux de donnees medicaux structurent les connaissances sous forme de questions a choix multiple ou de reponses attendues tres cadres.

Sur le texte libre, la lecture demande davantage de nuance. Le gain de `METEOR` est reel, ce qui suggere une meilleure proximite lexicale moyenne avec les references. La distance euclidienne evolue elle aussi dans le bon sens, avec une baisse de `0.2278` a `0.2231`, signe d'un rapprochement moyen de la distribution lexicale vers la reference. En revanche, la similarite cosinus reste plus reservee : la moyenne progresse legerement, mais la mediane recule et le modele de base reste meilleur sur davantage d'exemples individuels. Autrement dit, l'amelioration existe, mais elle n'est pas uniforme selon la metrique retenue.

Cette dissymetrie est assez coherente avec un SFT qui apprend bien des patrons de reponse attendus, mais qui ne garantit pas encore une amelioration stable de la qualite clinique discursive. Le POC montre donc une progression reelle, surtout sur les formats structures, sans permettre pour autant de conclure a une fiabilite suffisante pour un usage sensible ni a un gain semantique uniforme sur la generation libre.

### 5.3 Analyse critique et nuancee

L'un des points positifs du projet est justement de ne pas s'arreter a un simple tableau de scores. Cela n'empeche pas de garder plusieurs limites importantes bien visibles.

Premiere limite : les metriques utilisees restent imparfaites pour juger une reponse medicale. `METEOR`, la similarite cosinus, la distance euclidienne ou le scoring par premiere lettre sont utiles pour une premiere comparaison, mais ils ne mesurent pas directement la securite, la factualite, la nuance clinique ou l'adequation a un contexte patient reel.

Deuxieme limite : l'evaluation disponible porte sur le checkpoint SFT, pas encore sur un benchmark final DPO consolide dans le depot. Il reste donc trop tot pour conclure serieusement sur l'apport effectif de DPO dans ce projet.

Troisieme limite : l'augmentation de la variance reste un signal d'alerte. Le modele fine-tune semble plus specialise, mais aussi moins regulier, y compris sur certaines metriques texte pourtant plus favorables en moyenne. Pour un POC medical, ce point compte beaucoup : quelques gains marquants ne compensent pas automatiquement plusieurs echecs critiques.

Quatrieme limite : aucun echantillon commente de reussites et d'echecs n'est encore archive dans le rapport. C'est un manque important, car une lecture clinique exige une analyse qualitative des erreurs, par exemple sur les omissions de red flags, les formulations trop affirmatives ou les reponses plausibles mais inexactes.

La conclusion la plus juste est donc la suivante : le POC montre un signal d'efficacite encourageant, surtout sur les taches structurees, mais il ne permet pas encore de revendiquer une qualite clinique robuste sur la generation libre.

## 6. Analyse cout/performance

### 6.1 Performance d'inference : latence et debit

Le projet contient deja un dispositif de mesure dans `scripts/benchmark.py`, prevu pour benchmarker la latence et le debit du backend `vllm` a travers l'API FastAPI. Le script collecte notamment :

- la latence par requete en millisecondes ;
- la latence moyenne, mediane et p95 ;
- le nombre de tokens generes ;
- le debit en tokens par seconde.

Ce choix de mesure est pertinent, car il observe la performance au niveau applicatif reel et pas seulement au niveau du moteur vLLM. Il integre donc le cout de la couche HTTP et donne une vision plus proche d'un service effectivement deploye.

En revanche, le depot actuel ne contient pas encore de sortie benchmark archivee. A ce stade, le rapport peut donc decrire proprement la methode de mesure et l'infrastructure cible, mais pas encore presenter des chiffres finaux de latence ou de debit valides experimentalement.

### 6.2 Cout d'infrastructure pour le deploiement

La documentation GCP du projet recommande une VM `n1-standard-4` avec `1 x NVIDIA T4`, idealement en mode Spot. Le cout indicatif documente dans le depot est de l'ordre de `0,19 $ a 0,26 $ / heure` pour cette configuration frugale.

Sur cette base, on peut etablir un cadrage simple du cout de serving GPU :

| Duree d'utilisation GPU | Cout estime bas | Cout estime haut |
|---|---:|---:|
| 1 heure | 0,19 $ | 0,26 $ |
| 10 heures | 1,90 $ | 2,60 $ |
| 30 heures | 5,70 $ | 7,80 $ |
| 100 heures | 19,00 $ | 26,00 $ |

Il faut naturellement ajouter a cela le stockage persistant et, selon l'architecture retenue, le cout d'une petite instance CPU ou de quelques services reseau. Le projet rappelle d'ailleurs une bonne pratique tres simple : arreter la VM hors utilisation pour que la facturation du CPU et du GPU s'interrompe immediatement.

Cette strategie convient bien a un POC, avec un corollaire assez clair : tant que le service n'est pas automatise et benchmarke, le cout reste surtout pilote par le temps de machine allumee, et non encore par une capacite verifiee en requetes par seconde.

### 6.3 Cout d'infrastructure pour l'entrainement

Le besoin exprime porte aussi sur les heures GPU d'entrainement. Sur ce point, le depot documente la configuration d'entrainement et les artefacts produits, mais il n'archive pas dans le workspace actuel les journaux exploitables permettant de reconstituer de facon certaine :

- la duree SFT effective ;
- la duree DPO effective ;
- le nombre total d'heures GPU consommees ;
- le cout reel associe.

Il serait peu rigoureux d'inventer ces valeurs. Le plus propre est donc de distinguer deux niveaux :

- le cout reel observe, qui n'est pas encore consolide dans les fichiers disponibles ;
- le cout previsionnel, qui depend directement du nombre d'heures GPU effectivement consommees.

Avec la meme hypothese d'une machine GPU entre `0,19 $` et `0,26 $` par heure, le cout d'entrainement peut se lire de facon lineaire : `cout entrainement = heures GPU x tarif horaire GPU`.

Exemples de lecture :

| Heures GPU cumulees | Cout estime bas | Cout estime haut |
|---|---:|---:|
| 2 h | 0,38 $ | 0,52 $ |
| 5 h | 0,95 $ | 1,30 $ |
| 10 h | 1,90 $ | 2,60 $ |
| 20 h | 3,80 $ | 5,20 $ |

Ces ordres de grandeur montrent bien l'interet du couple `Qwen3-1.7B + LoRA + 4-bit` : le projet s'inscrit clairement dans une logique de fine-tuning et de serving a faible cout, surtout en comparaison de modeles plus grands ou d'un full fine-tuning.

### 6.4 Lecture cout/performance globale

Au stade actuel, le message technique le plus utile a retenir est le suivant :

- du cote qualite, le POC montre un gain tangible sur les taches structurees ;
- du cote cout, l'architecture choisie est volontairement frugale et compatible avec un budget etudiant ;
- du cote performance d'inference, la methode de benchmark existe deja mais les chiffres cibles restent a produire ;
- du cote entrainement, les choix LoRA et quantification 4-bit reduisent crediblement le cout, mais le total d'heures GPU doit encore etre consigne proprement.

En resume, la trajectoire cout/performance du projet parait saine, mais elle n'est pas encore completement demontree par des mesures d'exploitation archivees.

## 7. Limites du POC

Le projet assume deja plusieurs limites importantes, qu'il faut maintenir explicitement dans un rapport professionnel.

- Il s'agit d'un POC pedagogique, pas d'un dispositif medical.
- L'anonymisation reste heuristique et non suffisante pour un contexte de production.
- Les metriques automatiques ne suffisent pas a valider la surete clinique.
- Le backend `echo` de la CI ne valide pas l'inference reelle du modele fine-tune.
- Le pipeline de deploiement existe, mais le benchmark complet FastAPI + vLLM avec le vrai modele reste a finaliser.
- Les chiffres de latence, debit, VRAM et heures GPU ne sont pas encore centralises dans des artefacts de reporting versionnes.

Cette lucidite renforce plutot le rapport qu'elle ne l'affaiblit. Dans un contexte medical, ne pas surestimer les resultats releve de la rigueur, pas d'un aveu d'echec.

## 8. Roadmap priorisee

Les prochaines etapes les plus importantes pour transformer ce POC en dossier plus solide sont les suivantes :

1. executer `scripts/benchmark.py` sur une instance GPU avec le vrai backend `vllm` et archiver les resultats ;
2. exporter dans le depot un tableau de synthese avec latence moyenne, p95, tokens/s, VRAM et cout horaire ;
3. consigner les durees reelles SFT et DPO pour calculer proprement les heures GPU et le cout d'entrainement ;
4. ajouter une evaluation qualitative d'exemples reussis et d'echecs cliniquement significatifs ;
5. evaluer explicitement le checkpoint DPO pour verifier s'il stabilise les gains du SFT ;
6. rendre le deploiement Google Cloud reproductible de bout en bout avec IAM, build, push et deploiement documentes ;
7. ajouter des garde-fous applicatifs sur les demandes urgentes, dangereuses ou hors perimetre medical.

## 9. Conclusion

Le projet actuel constitue un POC credible de fine-tuning medical bilingue a faible cout. Dans le cadre retenu, les choix methodologiques restent globalement coherents : `Qwen3-1.7B` comme contrainte de depart mais socle encore defensable pour un POC frugal, `LoRA` et la quantification 4-bit pour contenir le cout d'entrainement, `DPO` pour explorer un alignement plus fin, et `vLLM` pour viser un serving plus performant qu'une integration naive.

Les resultats disponibles montrent un gain net sur les taches structurees, en particulier les QCM. Sur le texte libre, l'image est plus contrastee : `METEOR` et la distance euclidienne evoluent dans le bon sens en moyenne, tandis que la similarite cosinus reste plus ambigue. La progression est donc reelle, mais encore insuffisante pour parler de qualite clinique robuste.

Sur le plan cout/performance, l'architecture cible reste bien choisie pour un budget limite, mais le projet doit encore produire ses mesures finales de latence, debit et heures GPU pour transformer une intuition technique solide en demonstration complete. La suite logique n'est donc pas de changer radicalement d'approche, mais de fermer proprement la boucle de preuve experimentale : benchmark, couts reels, analyse qualitative des erreurs et deploiement reproductible.

