# Fine-tuning medical bilingue avec Qwen3, LoRA et DPO

Ce depot est un POC de fine-tuning medical bilingue francais/anglais autour de `Qwen3-1.7B`. Il couvre la preparation d'un dataset SFT/DPO, l'entrainement LoRA avec Unsloth/PEFT, une passe DPO et une evaluation comparative entre le modele de base et le modele fine-tune.

Etat actuel : la partie dataset et entrainement est avancee. La partie deploiement API, Docker, CI/CD et benchmark d'inference reste a finaliser.

## Objectifs

- Construire un dataset medical bilingue pour l'entrainement supervise.
- Produire un dataset DPO de preferences `chosen/rejected` pour l'alignement.
- Fine-tuner `Qwen3-1.7B` avec LoRA en 4-bit.
- Comparer le modele de base et le modele fine-tune sur un split de test.
- Documenter les limites, les risques et les prochaines etapes du POC.

## Structure du depot

```text
.
|-- README.md
|-- main.py
|-- pyproject.toml
|-- uv.lock
|-- recap.md
`-- notebooks/
    |-- hf_medical_datasets_eda.ipynb
    |-- colab_qwen3_unsloth_finetune.ipynb
    |-- colab_qwen3_unsloth_eval_compare.ipynb
    |-- qwen3-medical-lora/
    |-- qwen3-medical-dpo-lora/
    |-- sft_output/
    |-- dpo_output/
    `-- eval_results/
```

Fichiers principaux :

- `notebooks/hf_medical_datasets_eda.ipynb` : EDA, normalisation, construction des datasets SFT/DPO, anonymisation et controle PII.
- `notebooks/colab_qwen3_unsloth_finetune.ipynb` : entrainement SFT LoRA puis DPO avec Unsloth/TRL.
- `notebooks/colab_qwen3_unsloth_eval_compare.ipynb` : evaluation comparative entre le modele de base et le checkpoint SFT.
- `notebooks/qwen3-medical-lora/` : adapter LoRA SFT exporte.
- `notebooks/qwen3-medical-dpo-lora/` : adapter LoRA apres DPO.
- `notebooks/eval_results/` : generations et metriques d'evaluation.
- `recap.md` : audit honnete de l'avancement par rapport aux criteres de validation.

## Environnement

Le projet utilise `uv` et Python `>=3.13`.

Dependances principales declarees dans `pyproject.toml` :

- `datasets`
- `transformers`
- `trl`
- `peft`
- `accelerate`
- `bitsandbytes`
- `unsloth`
- `unsloth-zoo`
- `tensorboard`
- `presidio-analyzer`
- `presidio-anonymizer`
- `pandas`, `matplotlib`, `seaborn`

Installation :

```bash
uv sync
```

L'entrainement necessite un GPU CUDA. Les notebooks le verifient explicitement avec `torch.cuda.is_available()`.

## Sources de donnees

Les sources utilisees sont declarees dans `notebooks/hf_medical_datasets_eda.ipynb`.

| Source | Repo Hugging Face | Langue | Usage | Licence |
|---|---|---:|---|---|
| MediQAl | `ANR-MALADES/MediQAl` | fr | SFT, questions ouvertes et QCM | A verifier sur la fiche HF |
| frenchmedmcqa | `nthngdy/frenchmedmcqa` | fr | SFT, QCM medical | A verifier sur la fiche HF |
| MedQuad | `keivalya/MedQuad-MedicalQnADataset` | en | SFT, questions/reponses medicales | A verifier sur la fiche HF |
| UltraMedical-Preference | `TsinghuaC3I/UltraMedical-Preference` | en | DPO, paires de preference | A verifier sur la fiche HF |

Note : les fichiers `dataset_info.json` presents dans le cache local ne renseignent pas les champs `license`, `citation` ou `homepage`. Les licences doivent donc etre confirmees depuis les fiches Hugging Face avant rendu final.

## Datasets produits

Deux datasets finaux sont construits et consommes par les notebooks d'entrainement :

- `Maphe/medical-sft-5k`
- `Maphe/medical-dpo-5k`

### Dataset SFT

Objectif : entrainement supervise instruction/reponse.

Format : Hugging Face Dataset avec splits :

| Split | Nombre d'exemples cible |
|---|---:|
| `train` | 5 000 |
| `validation` | 500 |
| `test` | 500 |

Schema principal :

| Colonne | Description |
|---|---|
| `dataset` | source d'origine normalisee |
| `source_family` | famille de source |
| `source_repo_id` | repo Hugging Face source |
| `source_config` | configuration source si applicable |
| `split` | split source original |
| `source_id` | identifiant source ou identifiant reconstruit |
| `language` | `fr` ou `en` |
| `task_type` | `open_qa`, `mcq_single`, etc. |
| `topic` | sujet medical si disponible |
| `instruction` | prompt final d'entrainement |
| `response` | reponse attendue |
| `answer_key` | lettre de reponse pour les QCM |
| `answer_index` | index de reponse pour les QCM |

Le notebook applique une deduplication sur les paires textuelles `instruction/response`, puis un echantillonnage par quotas de sources.

### Dataset DPO

Objectif : alignement par preference clinique.

Source principale : `TsinghuaC3I/UltraMedical-Preference`.

Format :

| Colonne | Description |
|---|---|
| `prompt` | instruction utilisateur normalisee |
| `chosen` | reponse preferee |
| `rejected` | reponse rejetee |
| `language` | langue de l'exemple |
| `source_family` | famille source |
| `source_repo_id` | repo Hugging Face source |

Taille cible : `5 000` paires `prompt/chosen/rejected`.

## Preparation et conformite RGPD

Le pipeline de preparation est trace dans `notebooks/hf_medical_datasets_eda.ipynb`.

Etapes principales :

1. Chargement des sources Hugging Face dans `.cache/huggingface`.
2. Analyse exploratoire des schemas, colonnes texte, tailles et valeurs manquantes.
3. Normalisation des sources vers un schema commun.
4. Construction des instructions SFT.
5. Construction des paires DPO.
6. Deduplication des exemples.
7. Echantillonnage par quotas.
8. Split SFT `train/validation/test`.
9. Anonymisation heuristique.
10. Controle final des entites PII restantes.

L'anonymisation utilise `presidio-anonymizer` avec des detecteurs regex pour :

- `EMAIL`
- `PHONE`
- `URL`
- `DATE`
- `AGE`
- `PERSON`

Les valeurs detectees sont remplacees par des tags, par exemple `[EMAIL]` ou `[PERSON]`.

Limite importante : cette anonymisation est adaptee a un POC, mais elle reste heuristique. Pour un usage de production, il faudrait ajouter une revue plus stricte, des tests automatiques anti-PII et une validation juridique/metier.

## Modele et entrainement

Modele de base :

- `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`

Technique :

- LoRA via Unsloth/PEFT.
- Chargement 4-bit avec bitsandbytes.
- Entrainement SFT avec TRL `SFTTrainer`.
- Alignement DPO avec TRL `DPOTrainer`.

Parametres LoRA :

| Parametre | Valeur |
|---|---:|
| Rank `r` | 16 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0 |
| Bias | `none` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

Parametres SFT :

| Parametre | Valeur |
|---|---:|
| Epochs | 2 |
| Batch size par device | 32 |
| Gradient accumulation | 16 |
| Learning rate | `2e-4` |
| Scheduler | cosine |
| Warmup ratio | `0.03` |
| Optimiseur | `adamw_8bit` |
| Seed | 42 |
| Max sequence length | 1024 |
| Logging | TensorBoard |

Parametres DPO :

| Parametre | Valeur |
|---|---:|
| Epochs | 1 |
| Batch size par device | 4 |
| Gradient accumulation | 8 |
| Learning rate | `5e-5` |
| Beta | `0.1` |
| Scheduler | cosine |
| Warmup ratio | `0.1` |
| Optimiseur | `adamw_8bit` |
| Seed | 42 |
| Max sequence length | 1024 |

Artefacts produits :

- SFT : `notebooks/qwen3-medical-lora/`
- DPO : `notebooks/qwen3-medical-dpo-lora/`
- Checkpoints SFT : `notebooks/sft_output/checkpoint-*`
- Checkpoint DPO : `notebooks/dpo_output/checkpoint-157`

Les fichiers `adapter_config.json` confirment l'utilisation de PEFT LoRA avec `peft_type = LORA`, `r = 16` et `lora_alpha = 16`.

## Suivi d'experimentation

Les logs d'entrainement sont presents dans :

- `notebooks/sft_output/checkpoint-625/trainer_state.json`
- `notebooks/dpo_output/checkpoint-157/trainer_state.json`
- `notebooks/sft_output/tensorboard/`

Pour ouvrir TensorBoard depuis la racine du projet :

```bash
uv run tensorboard --logdir notebooks/sft_output/tensorboard --host 0.0.0.0 --port 8000
```

Observation rapide :

- La loss SFT passe d'environ `1.70` au debut a une zone autour de `0.9-1.1` en fin d'entrainement, avec des fluctuations normales.
- Les logs DPO contiennent notamment `rewards/accuracies`, `rewards/chosen`, `rewards/rejected` et `rewards/margins`.

## Evaluation

Notebook :

- `notebooks/colab_qwen3_unsloth_eval_compare.ipynb`

Resultats :

- `notebooks/eval_results/qwen3_base_vs_sft_output_summary.json`
- `notebooks/eval_results/qwen3_base_vs_sft_output.jsonl`
- `notebooks/eval_results/qwen3_base_vs_sft_output.csv`

Evaluation realisee sur `500` exemples.

| Metrique | Modele de base | Modele fine-tune SFT | Delta |
|---|---:|---:|---:|
| METEOR moyen texte libre | 0.1361 | 0.1653 | +0.0292 |
| Score QCM first-letter | 0.0515 | 0.4378 | +0.3863 |
| QCM corrects | 12 | 102 | +90 |

Lecture prudente :

- Le fine-tuning ameliore fortement les QCM dans cette evaluation.
- Le gain METEOR sur texte libre est positif mais plus modeste.
- METEOR et le scoring par premiere lettre restent des metriques limitees pour juger la qualite clinique. Une revue qualitative d'exemples reussis et echoues reste necessaire.

## Reproduction du workflow

Ordre conseille :

1. Ouvrir `notebooks/hf_medical_datasets_eda.ipynb`.
2. Charger les sources Hugging Face.
3. Construire les artefacts SFT/DPO.
4. Pousser ou verifier les datasets `Maphe/medical-sft-5k` et `Maphe/medical-dpo-5k`.
5. Ouvrir `notebooks/colab_qwen3_unsloth_finetune.ipynb`.
6. Lancer le SFT ou charger l'adapter SFT existant.
7. Lancer le DPO si necessaire.
8. Ouvrir `notebooks/colab_qwen3_unsloth_eval_compare.ipynb`.
9. Comparer le modele de base et le checkpoint fine-tune.

Les notebooks utilisent un `SEED = 42` pour stabiliser les splits, l'echantillonnage et l'entrainement.

## Securite, ethique et limites

Ce projet est un POC pedagogique. Le modele ne doit pas etre utilise comme dispositif medical ni comme substitut a un professionnel de sante.

Risques identifies :

- hallucinations medicales ;
- reponses trop affirmatives ;
- biais issus des datasets sources ;
- couverture incomplete des pathologies, populations et langues ;
- anonymisation imparfaite ;
- evaluation automatique insuffisante pour valider une qualite clinique.

Mesures deja presentes :

- system prompt demandant des reponses claires, factuelles et structurees ;
- anonymisation heuristique des donnees ;
- separation train/validation/test pour le SFT ;
- evaluation comparative sur un split de test.

Mesures a ajouter :

- tests de securite medicale ;
- exemples d'echecs commentes ;
- avertissement utilisateur dans l'API ;
- refus ou redirection en cas de demande urgente ou dangereuse ;
- revue humaine d'un echantillon de generations.

## Deploiement API

Le depot contient maintenant une API FastAPI minimale dans `app/` avec :

- `GET /health` : statut du service, backend actif et identifiant modele ;
- `POST /generate` : generation de texte via un backend d'inference ;
- logs JSON par requete, reponse et generation ;
- backend local `echo` par defaut pour valider le packaging sans GPU ;
- tests Pytest dans `tests/` ;
- Dockerfile et workflow GitHub Actions ;
- dependances API isolees dans `deployment/pyproject.toml`.

Le backend par defaut ne remplace pas le modele fine-tune. Il sert de fallback leger pour le POC local, la CI et les tests Docker. La justification du choix vLLM pour une cible production est documentee dans `docs/vllm.md`.

### Lancer l'API en local avec uv

```bash
uv run --project deployment --no-dev uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Exemple curl

```bash
curl -s http://localhost:8000/health
```

```bash
curl -s -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Quels sont les signes d alerte d une douleur thoracique ?","max_new_tokens":64,"temperature":0.2}'
```

### Tests

```bash
uv run --project deployment --group dev pytest -q
```

### Docker

```bash
docker build -t fine-tuning-oc-api .
docker run --rm -p 8000:8000 fine-tuning-oc-api
```

### CI/CD

Le workflow `.github/workflows/ci.yml` execute :

1. installation du projet API via `uv sync --project deployment --group dev` ;
2. import check avec `compileall` ;
3. tests Pytest ;
4. build Docker.

## Roadmap priorisee

1. Completer les licences exactes des sources dans ce README.
2. Ajouter une preuve anti-data-leakage entre `train`, `validation` et `test`.
3. Exporter les courbes TensorBoard en images dans un dossier `reports/`.
4. Transformer les notebooks critiques en scripts reproductibles :
   - `scripts/build_dataset.py`
   - `scripts/train_sft.py`
   - `scripts/train_dpo.py`
   - `scripts/evaluate.py`
5. Ajouter une API d'inference avec logs auditables.
6. Ajouter Docker et tests Pytest.
7. Ajouter une CI GitHub Actions.
8. Rediger un rapport technique avec cout GPU, latence, debit, analyse critique et recommandations de passage a l'echelle.

## Versioning

Le dossier de travail contient les artefacts de preparation, d'entrainement et d'evaluation. Les fichiers sont destines a etre versionnes dans le depot du projet, et les datasets/modeles sont references via Hugging Face Hub.

Pour un rendu final, ajouter ici :

- URL du depot Git ;
- commit du rendu ;
- revisions Hugging Face des datasets ;
- revisions Hugging Face ou chemins finaux des adapters LoRA.
