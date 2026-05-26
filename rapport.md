# Rapport d'analyse des resultats SFT

## Contexte

Ce rapport synthese analyse les resultats du fichier `notebooks/eval_results/qwen3_base_vs_sft_output_summary.json`.

- Dataset evalue : `Maphe/medical-sft-5k`
- Modele de base : `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`
- Checkpoint fine-tune evalue : `sft_output/checkpoint-625`
- Nombre total d'exemples : `500`

La comparaison porte sur deux familles de taches :

- `267` exemples de texte libre
- `233` exemples de QCM

## Synthese executive

Le fine-tuning SFT ameliore globalement les performances du modele, mais de facon tres differente selon le type de tache.

Le gain le plus net concerne les QCM. Le modele fine-tune obtient `102` reponses correctes sur `233`, contre seulement `12` pour le modele de base. Cela correspond a un passage de `5,2 %` a `43,8 %` de bonnes reponses. Sur cette dimension, l'amelioration est forte et sans ambiguite.

Sur le texte libre, les resultats progressent aussi, mais de maniere plus moderee et moins reguliere. Les metriques lexicales et certaines mesures semantiques s'ameliorent en moyenne, mais le modele fine-tune reste plus instable que le modele de base. En pratique, cela signifie qu'il produit davantage de bonnes reponses, mais avec une variabilite plus forte selon les exemples.

En l'etat, le fine-tuning semble surtout avoir ameliore l'alignement du modele avec les taches structurees, tout en apportant un gain plus partiel sur la generation libre.

## Resultats cles

### 1. Texte libre

Sur les `267` exemples de texte libre, la metrique `METEOR` passe de `0.1361` a `0.1653`, soit un gain absolu de `+0.0292`. Cela represente une hausse relative d'environ `+21,4 %`. Ce signal est positif et indique une meilleure proximite entre les reponses generees et les references attendues.

Le detail des victoires confirme cette tendance :

- Fine-tune meilleur sur `METEOR` : `139` cas
- Base meilleur sur `METEOR` : `115` cas
- Egalites : `13` cas

La distance euclidienne s'ameliore egalement, de `0.2278` a `0.2231`. Comme une valeur plus faible est meilleure, cela va dans le sens d'une progression semantique legere mais coherente. Sur cette metrique, le modele fine-tune est meilleur dans `159` cas contre `108` pour le modele de base.

La similarite cosinus donne un signal plus mitige. La moyenne augmente legerement, de `0.3762` a `0.3859`, mais la mediane baisse de `0.4167` a `0.3917`. De plus, le modele de base gagne plus souvent sur cette mesure :

- Fine-tune meilleur sur la similarite cosinus : `113` cas
- Base meilleur sur la similarite cosinus : `141` cas
- Egalites : `13` cas

Cette combinaison suggere que le fine-tuning apporte de gros gains sur certains exemples, mais qu'il degrade encore une partie non negligeable des reponses libres.

### 2. QCM

Les QCM sont la zone de progression la plus claire.

Le `qcm_score` moyen passe de `0.0515` a `0.4378`, soit un gain absolu de `+0.3863`.

En nombre de bonnes reponses exactes :

- Modele de base : `12 / 233`
- Modele fine-tune : `102 / 233`

Le modele fine-tune multiplie donc le nombre de bonnes reponses par environ `8,5`. Ce resultat montre que le SFT a fortement ameliore la capacite du modele a identifier ou formater la bonne option de reponse.

La comparaison `qcm_first_letter` va dans le meme sens :

- Fine-tune meilleur : `93` cas
- Base meilleur : `3` cas
- Egalites : `137` cas

Le nombre eleve d'egalites montre qu'une partie importante des cas reste non discriminante, ou que les deux modeles echouent encore souvent a produire exactement le format attendu. Malgre cela, l'avantage du modele fine-tune reste tres net.

### 3. Classement global

Le score global `rank_1` favorise aussi le modele fine-tune :

- Base `rank_1` : `269`
- Fine-tune `rank_1` : `368`
- Egalites : `137`

En retirant les egalites, le modele fine-tune est devant dans `231` cas contre `132` pour le modele de base, soit environ `63,6 %` des cas decides. Cela confirme un avantage global du modele fine-tune sur l'ensemble de l'evaluation.

## Stabilite des performances

Un point important de cette evaluation est l'augmentation de la variabilite apres fine-tuning.

Les ecarts-types montent sur plusieurs metriques :

- `METEOR` : `0.0916` -> `0.1528`
- Similarite cosinus : `0.1971` -> `0.2506`
- `QCM score` : `0.2215` -> `0.4972`

Cela signifie que le modele fine-tune est plus heterogene. Il reussit mieux certaines generations, parfois nettement, mais il reste moins regulier que le modele de base. D'un point de vue pratique, cela traduit souvent un modele plus specialise sur la tache, mais pas encore pleinement stabilise.

## Longueur des reponses

La longueur moyenne des sorties varie peu :

- Base : `283.95` caracteres
- Fine-tune : `286.73` caracteres

En revanche, la mediane baisse :

- Base : `389`
- Fine-tune : `315.5`

Le modele fine-tune tend donc a produire des reponses typiquement plus courtes, tout en conservant quelques sorties longues qui maintiennent la moyenne. Ce comportement peut etre coherent avec un apprentissage plus directif ou plus centre sur la consigne.

## Interpretation

Les resultats montrent que le fine-tuning a eu un impact reel et utile.

- Pour les QCM, le gain est fort, clair et directement exploitable.
- Pour le texte libre, le gain existe, mais il reste partiel et plus instable.
- Le modele fine-tune semble mieux aligne avec les formats de taches attendus, surtout quand la sortie correcte est contrainte.

En d'autres termes, le fine-tuning ameliore nettement la precision sur les taches structurees et apporte une progression credible, mais encore imparfaite, sur les reponses libres.

## Conclusion

Le checkpoint `sft_output/checkpoint-625` surpasse globalement le modele de base sur l'evaluation realisee. L'amelioration est particulierement forte sur les QCM, ou le modele fine-tune passe de `12` a `102` bonnes reponses sur `233` exemples. Sur le texte libre, les gains sont reels mais plus modestes, avec des signaux positifs sur `METEOR` et la distance euclidienne, contrebalances par une similarite cosinus plus irreguliere.

La conclusion la plus honnete est donc la suivante : le SFT a bien ameliore l'alignement du modele avec les taches medicales ciblees, surtout pour les taches structurees, mais il reste du travail pour rendre les gains sur le texte libre plus robustes et plus reguliers.

## Recommandations

1. Produire des graphiques separes pour les QCM et le texte libre afin d'eviter une lecture trop globale des moyennes.
2. Inspecter manuellement un echantillon de cas ou le modele de base reste meilleur sur la similarite cosinus.
3. Ajouter une evaluation qualitative humaine sur les reponses libres medicales.
4. Completer l'analyse avec des intervalles de confiance ou un test statistique simple.
5. Evaluer aussi le checkpoint DPO pour verifier si l'alignement par preference stabilise les gains observes.