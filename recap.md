# Recap honnete de l'etat du projet

Date d'audit : 2026-05-18  
Depot audite : dossier de travail `fine_tuning_oc`

## Lecture rapide

Le projet est avance sur la partie **dataset medical bilingue** et **fine-tuning SFT/DPO avec LoRA**. Les notebooks montrent une vraie demarche de preparation, anonymisation, entrainement, logs TensorBoard et evaluation comparative.

En revanche, les livrables sont encore trop "dossier de travail" : beaucoup de preuves sont dans des notebooks et artefacts, mais il manque de la documentation racine, des scripts propres, un service API, Docker, CI/CD, tests, logs d'inference et rapport technique structure. Pour un evaluateur, le risque principal n'est pas que rien n'ait ete fait, mais que les preuves soient dispersees ou absentes.

Synthese estimee :

| Competence / livrable | Etat |
|---|---|
| Dataset medical bilingue | Partiellement valide, plutot solide techniquement |
| Modele IA optimise | Partiellement valide, preuves SFT/DPO presentes |
| Endpoint de demonstration | Non valide en l'etat |
| Pipeline CI/CD | Non valide en l'etat |
| Rapport technique infra | Non valide en l'etat, a rediger |

## 1. Competence : Ajuster les parametres d'entrainement

### Livrable : Dataset medical bilingue

#### Corpus structure, documente et versionne

**Etat : partiel.**

Preuves presentes :

- Le notebook `notebooks/hf_medical_datasets_eda.ipynb` construit un dataset SFT Hugging Face avec splits `train`, `validation`, `test`.
- Les IDs Hugging Face utilises sont visibles dans le notebook d'entrainement :
  - `Maphe/medical-sft-5k`
  - `Maphe/medical-dpo-5k`
- Le format final attendu est bien Hugging Face Dataset, pas seulement des CSV ad hoc.
- Le notebook cible 5 000 lignes train SFT, 500 validation et 500 test, soit 6 000 exemples SFT au total.

Points faibles :

- Le `README.md` racine est vide.
- Je ne vois pas de README dataset local qui decrit clairement le schema, les sources, les licences et le processus de creation.
- Les sources sont listees dans le code, mais leurs licences ne sont pas documentees dans un livrable lisible.
- Le versioning est probablement fait via commits et/ou Hugging Face Hub, mais il faut le rendre explicite : lien du repo, revision/commit HF, date d'export, nom exact des datasets.

Verdict honnete : le dataset existe probablement et il est structure, mais la preuve documentaire demandee est insuffisante.

#### Processus de traitement tracable et conforme

**Etat : partiel a bon.**

Preuves presentes :

- `notebooks/hf_medical_datasets_eda.ipynb` contient les etapes de chargement, EDA, normalisation, deduplication, construction SFT/DPO, anonymisation et controle PII.
- Utilisation de `presidio-anonymizer` avec detection regex pour `EMAIL`, `PHONE`, `URL`, `DATE`, `AGE`, `PERSON`.
- Le notebook genere un `rgpd_report_df` avec comptage des tags PII restants.

Points faibles :

- La justification RGPD reste courte et implicite.
- Il faut ajouter une section ecrite : nature des donnees, base de conformite, minimisation, anonymisation/pseudonymisation, limites de Presidio/regex, absence de donnees patient identifiantes visee, controle final.
- L'anonymisation est heuristique : c'est acceptable pour un POC, mais il faut l'assumer.

Verdict honnete : la tracabilite technique est bonne, la justification RGPD doit etre formalisee.

#### Qualite et pertinence des jeux de donnees

**Etat : plutot bon, avec reserves.**

Preuves presentes :

- SFT :
  - cible `5 000` exemples train ;
  - `500` validation ;
  - `500` test ;
  - sources FR/EN : MediQAl, frenchmedmcqa, MedQuad.
- DPO :
  - construit depuis `TsinghuaC3I/UltraMedical-Preference`;
  - format `prompt`, `chosen`, `rejected`;
  - cible `5 000` paires.
- Deduplication des paires textuelles.
- Equilibrage source/type via quotas.

Points faibles :

- La notion de "haute qualite" doit etre prouvee par des exemples inspectes, criteres de filtrage, statistiques et limites.
- Le DPO est principalement anglophone d'apres le notebook ; c'est pertinent pour l'alignement clinique, mais moins bilingue que le SFT.
- Il faut documenter pourquoi ces sources sont adaptees au medical et quelles limites elles ont.

Verdict honnete : techniquement convaincant, mais il manque une synthese qualite dans un README/rapport.

#### Partitionnement train / validation / test sans fuite

**Etat : partiel.**

Preuves presentes :

- Fonction `split_sft_frame` dans le notebook.
- Splits cibles explicites : `train=5000`, `validation=500`, `test=500`.
- Split stratifie par `task_type`, `dataset`, `language`.
- Deduplication avant echantillonnage.

Points faibles :

- Il manque un test ou tableau prouvant l'absence de fuite entre splits, par exemple intersection vide sur `source_id` et/ou hash normalise de `instruction + response`.
- Le DPO n'a pas de split train/validation/test clairement expose dans les preuves locales.

Verdict honnete : la methode est raisonnable, mais la preuve anti-leakage doit etre ajoutee.

### Livrable : Modele IA optimise

#### Code d'entrainement propre, commente et reproductible

**Etat : partiel.**

Preuves presentes :

- `notebooks/colab_qwen3_unsloth_finetune.ipynb` contient SFT et DPO.
- Seeds documentees dans le code : `SEED = 42`, avec `random`, `numpy`, `torch`.
- Hyperparametres documentes dans le notebook :
  - modele de base : `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`
  - LoRA rank `16`, alpha `16`, dropout `0`
  - SFT : batch `32`, grad accumulation `16`, LR `2e-4`, epochs `2`
  - DPO : batch `4`, grad accumulation `8`, LR `5e-5`, beta `0.1`, epochs `1`
- Artefacts LoRA presents :
  - `notebooks/qwen3-medical-lora/`
  - `notebooks/qwen3-medical-dpo-lora/`

Points faibles :

- Les scripts SFT/DPO ne sont pas fournis comme scripts Python propres ; tout est dans un notebook.
- Le notebook est utile, mais pour le critere "scripts fournis", il faudrait ajouter `scripts/train_sft.py` et `scripts/train_dpo.py` ou documenter explicitement que le notebook est le script executable.
- Le `README.md` racine ne donne aucune commande de reproduction.

Verdict honnete : reproductible pour toi, pas encore assez packagé pour un evaluateur.

#### Techniques d'optimisation correctement mises en oeuvre

**Etat : bon.**

Preuves presentes :

- Usage Unsloth/PEFT LoRA via `FastModel.get_peft_model`.
- Target modules coherents pour Qwen : `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- `adapter_config.json` confirme `peft_type = LORA`, `r = 16`, `lora_alpha = 16`.
- Quantization 4-bit via modele `unsloth-bnb-4bit`.

Points faibles :

- Il faut expliquer dans un rapport pourquoi LoRA/4-bit/Unsloth ont ete choisis.
- Les logs DPO montrent des variations de loss/rewards ; il faut les interpreter prudemment.

Verdict honnete : l'implementation LoRA est credible.

#### Logs de convergence et suivi d'experimentations

**Etat : partiel a bon.**

Preuves presentes :

- Logs SFT dans `notebooks/sft_output/checkpoint-625/trainer_state.json`.
- TensorBoard events presents dans `notebooks/sft_output/tensorboard/`.
- SFT : loss passe environ de `1.70` au debut a autour de `0.9-1.1` en fin de run, avec fluctuations mais pas d'explosion.
- DPO : logs dans `notebooks/dpo_output/checkpoint-157/trainer_state.json`.
- DPO : `rewards/accuracies` souvent autour de `0.65-0.84` selon les steps.
- Notebook d'evaluation : `notebooks/colab_qwen3_unsloth_eval_compare.ipynb`.
- Resultats comparatifs : `notebooks/eval_results/qwen3_base_vs_sft_output_summary.json`.

Resultats d'evaluation trouves :

- `eval_rows = 500`
- METEOR texte libre :
  - base `0.1361`
  - fine-tuned `0.1653`
  - delta `+0.0292`
- QCM first-letter :
  - base `0.0515`
  - fine-tuned `0.4378`
  - delta `+0.3863`
- Victoires QCM :
  - base OK `12`
  - fine-tuned OK `102`

Points faibles :

- Pas de dashboard exporte ou capture visible dans le dossier.
- Les courbes TensorBoard existent, mais il faudrait les exporter en PNG ou les presenter dans un rapport.
- Il manque une analyse qualitative d'exemples reussis/echecs.

Verdict honnete : les donnees de suivi existent, mais la presentation livrable manque.

#### Securite et ethique du modele

**Etat : insuffisant.**

Preuves presentes :

- Le system prompt demande des reponses medicales claires/factuelles.
- Le notebook traite l'anonymisation des donnees.

Points faibles :

- Je ne vois pas de section dediee aux limites cliniques, hallucinations, biais, risques d'usage, non-remplacement d'un professionnel de sante.
- Pas de tests de securite medicale ou refus/avertissements.

Verdict honnete : a rediger imperativement.

## 2. Competence : Automatiser le deploiement

### Livrable : Endpoint de demonstration

#### Code de deploiement package et portable

**Etat : non valide en l'etat.**

Preuves presentes :

- `pyproject.toml` existe avec dependances ML.

Points faibles :

- `main.py` contient seulement un `Hello from fine-tuning-oc!`.
- Aucun fichier FastAPI/serveur d'inference trouve.
- Aucun Dockerfile trouve.
- Aucun fichier `docker-compose.yml` ou config de service trouve.
- Pas de structure package claire pour le service.

Verdict honnete : la partie endpoint reste a construire.

#### Service fonctionnel accessible via API

**Etat : non valide en l'etat.**

Points faibles :

- Pas d'API locale detectee.
- Pas d'exemple `curl`.
- Pas de route `/generate`, `/health` ou similaire.
- Pas de preuve de demonstration live.

Verdict honnete : critere non rempli.

#### Deploiement optimise pour l'inference LLM

**Etat : non valide en l'etat.**

Points faibles :

- Aucune integration vLLM detectee.
- Aucune justification vLLM dans la doc.
- Aucune metrique latence/debit d'inference exportee.
- Aucun test Pytest/CI detecte.

Verdict honnete : critere non rempli.

#### Tracabilite des interactions

**Etat : non valide en l'etat.**

Points faibles :

- Pas de service, donc pas de logs requete/reponse.
- Pas de format de log auditable.

Verdict honnete : critere non rempli.

### Livrable : Pipeline CI/CD

**Etat : non valide en l'etat.**

Points faibles :

- Aucun workflow `.github/workflows` detecte.
- Aucun test Pytest detecte.
- Aucun pipeline Docker/build/deploy detecte.

Verdict honnete : a faire quasiment de zero.

## 3. Competence : Evaluer l'infrastructure sous-jacente

### Rapport technique

#### Analyse cout/performance

**Etat : insuffisant.**

Preuves presentes :

- Les notebooks impriment/collectent certains temps d'evaluation et infos GPU.
- Les resultats d'evaluation modele base vs fine-tune sont exportes.

Points faibles :

- Pas de rapport structure trouve.
- Pas d'analyse cout GPU, heures d'entrainement, cout estime cloud/local.
- Pas de benchmark latence/debit API.

Verdict honnete : donnees partielles possibles, rapport a rediger.

#### Structure professionnelle du document

**Etat : non valide.**

Points faibles :

- Pas de `rapport_technique.md` ou document equivalent trouve.
- Le `README.md` est vide.

Verdict honnete : a produire.

#### Justification des choix methodologiques

**Etat : partiel.**

Preuves presentes :

- Les choix sont visibles dans le code : Qwen3-1.7B, LoRA, DPO, Unsloth, TensorBoard.

Points faibles :

- Les raisons ne sont pas exposees dans un document.
- Il faut justifier Qwen3-1.7B, LoRA, DPO, Unsloth, vLLM et les compromis VRAM/cout/performance.

Verdict honnete : le fond existe, la justification ecrite manque.

#### Analyse critique des resultats

**Etat : partiel faible.**

Preuves presentes :

- Evaluation quantitative base vs SFT disponible.
- Les gains QCM sont nets dans le resume JSON.

Points faibles :

- Pas d'analyse qualitative des echecs.
- Pas de discussion des limites METEOR, QCM first-letter, taille du test, biais de dataset.
- Pas d'analyse du DPO apres alignement final visible.

Verdict honnete : les chiffres existent, l'analyse reste a faire.

#### Recommandations d'optimisation / roadmap

**Etat : non valide.**

Points faibles :

- Pas de roadmap priorisee trouvee.
- Pas de recommandations concretes issues du POC.

Verdict honnete : a rediger.

## Priorites recommandees

### Priorite 1 : rendre les preuves lisibles

1. Remplir `README.md` racine avec :
   - objectif du projet ;
   - structure du repo ;
   - datasets HF ;
   - modeles/adapters produits ;
   - commandes de reproduction ;
   - limites medicales.
2. Ajouter un README dataset ou une section dataset :
   - schema SFT/DPO ;
   - sources et licences ;
   - processus de creation ;
   - anonymisation/RGPD ;
   - splits et anti-leakage.
3. Exporter les courbes TensorBoard ou screenshots dans un dossier `reports/figures/`.

### Priorite 2 : solidifier les criteres dataset/modele

1. Ajouter un test anti-fuite entre splits.
2. Ajouter un petit rapport qualite dataset avec exemples et statistiques.
3. Transformer le notebook d'entrainement en scripts :
   - `scripts/train_sft.py`
   - `scripts/train_dpo.py`
   - eventuellement `scripts/build_dataset.py`
4. Documenter les hyperparametres finaux dans un fichier stable, par exemple `configs/training.yaml`.

### Priorite 3 : construire le livrable deploiement

1. Creer une API FastAPI minimale :
   - `/health`
   - `/generate`
   - logs JSON par requete/reponse.
2. Ajouter un Dockerfile.
3. Ajouter un exemple `curl`.
4. Ajouter des tests Pytest basiques.
5. Ajouter une CI GitHub Actions :
   - installation ;
   - lint ou import check ;
   - tests Pytest ;
   - build Docker si possible.
6. Ajouter une justification vLLM, meme si le POC local utilise un fallback plus leger.

### Priorite 4 : produire le rapport technique

Plan conseille :

1. Introduction et objectif du POC.
2. Donnees et conformite.
3. Methodologie SFT/DPO.
4. Hyperparametres et infrastructure.
5. Resultats quantitatifs.
6. Analyse qualitative : succes/echecs.
7. Limites, risques, biais, hallucinations.
8. Cout/performance : training + inference.
9. Roadmap priorisee.

## Conclusion honnete

Le coeur ML est bien avance : preparation dataset, LoRA SFT, DPO, logs et evaluation comparative sont presents. Pour la competence "ajuster les parametres d'entrainement", tu es probablement proche d'un livrable defendable, a condition de mieux documenter et de prouver les points sensibles.

Pour la competence "automatiser le deploiement", l'etat actuel ne suffit pas : il manque l'API, Docker, tests, CI/CD, vLLM ou justification, logs et metriques d'inference.

Pour la competence "evaluer l'infrastructure", il faut transformer les observations techniques en rapport structure, critique et chiffre. Le projet a de la matiere, mais pas encore le livrable final.
