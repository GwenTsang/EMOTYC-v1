# EMOTYC v1

EMOTYC v1 est un package minimal pour prédire, entraîner une tête linéaire et
évaluer des labels multi-labels à partir de fichiers XLSX. La v1 couvre
l'inférence avec des bundles de modèles déjà entraînés, l'entraînement linéaire
sur features TF-IDF ou backbone ONNX, l'évaluation sur colonnes binaires et la
comparaison de résultats.

## Installation

Depuis la racine du dépôt :

```bash
pip install -e ".[dev]"
```

Pour utiliser `train_linear.py`, installez aussi les dépendances
d'entraînement :

```bash
pip install -e ".[train]"
```

ONNX Runtime est installé séparément afin d'éviter d'installer à la fois la
version CPU et la version GPU :

```bash
python scripts/install_onnxruntime.py --cpu
```

Pour une machine GPU, utilisez explicitement :

```bash
python scripts/install_onnxruntime.py --gpu
```

L'option `--auto` choisit la version CPU si aucun signal GPU/CUDA n'est détecté.

## Modèles et datasets disponibles

Les alias sont définis dans `src/emotyc/registry.py`.

### Modèles

| Alias | Dépôt Hugging Face |
|---|---|
| `emotyc_1` | `GwendalTsang/EMOTYC_1` |
| `emotyc_2` | `GwendalTsang/EMOTYC_2` |

### Datasets

| Alias | Dépôt Hugging Face | Fichier |
|---|---|---|
| `CyberAgg` | `GwendalTsang/CyberAggAdo` | `CyberAdoAgg_gold_global_total_latest.xlsx` |
| `ttk` | `GwendalTsang/TTK` | `emotexttokids_gold_flat.xlsx` |

Les modèles et datasets sont téléchargés automatiquement depuis Hugging Face
Hub lors de la première utilisation.

## Prédiction

`predict.py` lit un XLSX avec une colonne `TEXT`, applique le template choisi,
charge le modèle, puis affiche les labels prédits ligne par ligne.

Prédiction avec le modèle `emotyc_1` sur le dataset `ttk` :

```bash
python predict.py \
  --model emotyc_1 \
  --data ttk \
  --template raw \
  --threshold 0.5 \
  --batch-size 32 \
  --out predictions.xlsx
```

Prédiction avec le modèle `emotyc_2` sur le dataset `CyberAgg` :

```bash
python predict.py \
  --model emotyc_2 \
  --data CyberAgg \
  --template raw \
  --threshold 0.5 \
  --out predictions_cyberagg.xlsx
```

`--model` accepte un alias connu (`emotyc_1`, `emotyc_2`) ou le chemin local
d'un bundle v1. `--data` accepte un alias de dataset connu (`ttk`, `CyberAgg`)
ou le chemin d'un fichier XLSX local. `--out` est optionnel et écrit un XLSX
avec `TEXT`, les colonnes `pred_<label>` et les colonnes `proba_<label>`.

### Templates

Templates disponibles :

- `raw` : texte inchangé ;
- `bca` : format `before:</s>current: <texte></s>after:</s>`.

Avec `--template bca --use-context`, les lignes voisines du XLSX sont injectées
comme contexte immédiat : la ligne `i-1` dans `before:` et la ligne `i+1` dans
`after:`. N'utilisez cette option que si l'ordre des lignes représente de vraies
phrases adjacentes ; sur un échantillon mélangé ou non contigu, le contexte
devient artificiel.

```bash
python predict.py \
  --model emotyc_1 \
  --data ttk \
  --template bca \
  --use-context
```

## Évaluation

`evaluate.py` utilise le même chemin d'inférence que `predict.py`, puis compare
les prédictions aux colonnes labels présentes dans le XLSX.

### Évaluation sur un dataset connu avec `--dataset`

```bash
python evaluate.py \
  --model emotyc_1 \
  --dataset ttk \
  --template raw \
  --threshold 0.5 \
  --batch-size 32 \
  --save-config \
  --save-metrics \
  --save-predictions \
  --out-dir runs/eval_emotyc1_ttk
```

### Évaluation sur un dataset connu avec `--data`

Le flag `--data` accepte aussi un alias de dataset connu :

```bash
python evaluate.py \
  --model emotyc_2 \
  --data CyberAgg \
  --template raw \
  --save-metrics \
  --out-dir runs/eval_emotyc2_cyberagg
```

### Évaluation avec template BCA et contexte

```bash
python evaluate.py \
  --model emotyc_1 \
  --dataset CyberAgg \
  --template bca \
  --use-context \
  --save-metrics \
  --save-predictions \
  --out-dir runs/eval_emotyc1_cyberagg_bca
```

### Évaluation sur un fichier XLSX local

```bash
python evaluate.py \
  --model emotyc_1 \
  --data mon_fichier_annote.xlsx \
  --template raw \
  --save-metrics \
  --out-dir runs/eval_local
```

> **Note :** `--dataset` et `--data` sont mutuellement exclusifs. Utilisez
> `--dataset` pour un alias de dataset connu, ou `--data` pour un fichier XLSX
> local ou un alias de dataset.

Les sorties optionnelles sont :

- `config.json` : configuration de la commande ;
- `metrics.json` : micro-F1, macro-F1, exact match et métriques par label ;
- `predictions.xlsx` : texte, labels gold, prédictions et probabilités.

Les métriques v1 sont calculées avec NumPy : précision, recall, F1, support,
micro-F1, macro-F1 et exact match.

## Comparaison

`compare_results.py` compare deux fichiers `metrics.json` produits par
`evaluate.py`.

Commencez par produire deux évaluations avec `--save-metrics` :

```bash
python evaluate.py \
  --model emotyc_1 \
  --dataset ttk \
  --save-metrics \
  --out-dir runs/eval_emotyc1_ttk

python evaluate.py \
  --model emotyc_2 \
  --dataset ttk \
  --save-metrics \
  --out-dir runs/eval_emotyc2_ttk
```

Puis comparez les résultats :

```bash
python compare_results.py \
  runs/eval_emotyc1_ttk/metrics.json \
  runs/eval_emotyc2_ttk/metrics.json \
  --out comparaison_ttk.json
```

La comparaison contient les écarts de métriques globales, les écarts par labels
communs, les labels présents seulement dans A et les labels présents seulement
dans B.

## Entraînement linéaire

`train_linear.py` entraîne une tête `LinearSVC` par label à partir d'un XLSX
annoté. L'encodeur peut être :

- `tfidf` : features TF-IDF apprises sur le fichier d'entraînement ;
- `onnx` : embeddings extraits avec un backbone ONNX existant.

### Entraînement TF-IDF sur le dataset `ttk`

```bash
python train_linear.py train \
  --data ttk \
  --labels joie tristesse colere \
  --encoder tfidf \
  --out-dir runs/linear_tfidf_ttk \
  --save-predictions
```

### Entraînement TF-IDF sur le dataset `CyberAgg`

```bash
python train_linear.py train \
  --data CyberAgg \
  --labels joie tristesse colere \
  --encoder tfidf \
  --out-dir runs/linear_tfidf_cyberagg \
  --save-predictions
```

### Entraînement avec un backbone ONNX

`--backbone-model` accepte soit un alias de modèle EMOTYC (`emotyc_1`,
`emotyc_2`), soit un dossier backbone-only contenant `tokenizer.json` et
`model.onnx` ou `backbone.onnx`.

```bash
python train_linear.py train \
  --data ttk \
  --labels joie tristesse colere \
  --encoder onnx \
  --backbone-model emotyc_1 \
  --out-dir runs/linear_onnx_ttk
```

### Évaluation d'un modèle linéaire sauvegardé

```bash
python train_linear.py evaluate \
  --model runs/linear_tfidf_ttk \
  --data ttk \
  --save-metrics \
  --save-predictions \
  --out-dir runs/linear_eval_ttk
```

### Options avancées de `train`

| Option | Défaut | Description |
|---|---|---|
| `--max-features` | aucun | Nombre maximal de features TF-IDF |
| `--ngram-min` | `1` | Borne basse des n-grams TF-IDF |
| `--ngram-max` | `2` | Borne haute des n-grams TF-IDF |
| `--no-lowercase` | désactivé | Désactiver la normalisation lowercase TF-IDF |
| `--c` | `1.0` | Paramètre C de LinearSVC |
| `--class-weight` | `none` | Pondération des classes (`none` ou `balanced`) |
| `--max-iter` | `1000` | Nombre maximal d'itérations LinearSVC |

Les artefacts produits par `train_linear.py train` sont :

- `linear_config.json` : labels, type d'encodeur et paramètres ;
- `linear_model.pkl` : vectorizer TF-IDF éventuel et classifieurs linéaires ;
- `backbone.onnx` et `tokenizer.json` pour les modèles entraînés avec
  `--encoder onnx` ;
- `train_metrics.json` et `training_config.json` ;
- `train_predictions.xlsx` si `--save-predictions` est utilisé.

Les scores `LinearSVC` sont transformés par sigmoid pour appliquer le seuil
global. Ces valeurs ne sont pas des probabilités calibrées.

## Format des bundles modèles

Un bundle v1 local est un dossier contenant exactement les artefacts nécessaires
à l'inférence :

- `backbone.onnx` : backbone ONNX ;
- `tokenizer.json` : tokenizer compatible `tokenizers` ;
- `head.npz` : poids de la tête de classification ;
- `head.json` : schéma de la tête ;
- `model_config.json` : labels du modèle.

`head.npz` doit contenir :

- `classifier.dense.weight`
- `classifier.dense.bias`
- `classifier.out_proj.weight`
- `classifier.out_proj.bias`

`head.json` décrit une tête v1 avec pooling sur `last_hidden_state[:, 0, :]`,
une couche dense avec activation `tanh`, une projection de sortie linéaire et
une activation sigmoid pour les probabilités.

L'encodage ONNX utilise `add_special_tokens=False`, conformément au format
EMOTYC/BCA : en template `bca`, la position 0 correspond au premier sous-mot de
`before` plutôt qu'à un token spécial ajouté. Pour limiter le padding et éviter
les pics mémoire GPU à grands batchs, les textes sont batchés par longueur
tokenisée puis les features sont remises dans l'ordre d'origine avant la tête de
classification.

`model_config.json` doit contenir soit :

```json
{"labels": ["label_a", "label_b"]}
```

soit :

```json
{"id2label": {"0": "label_a", "1": "label_b"}}
```

## Format des XLSX

Pour `predict.py`, le fichier doit contenir :

- `TEXT` : texte à prédire.

Pour `evaluate.py`, le fichier doit contenir :

- `TEXT` : texte à évaluer ;
- une ou plusieurs colonnes labels portant les mêmes noms que les labels du
  modèle.

Les colonnes labels doivent être binaires (`0` ou `1`). Seule l'intersection
entre les labels du modèle et les colonnes du XLSX est évaluée. Si aucune
colonne label commune n'est trouvée, l'évaluation s'arrête avec une erreur.

## Limites v1

- Pas de calibration.
- Pas de seuils par label.
- Pas de groupes ou splits dataset.
- Pas de sanity checks avancés.
- `scikit-learn` est requis seulement pour `train_linear.py`.
- ONNX Runtime est installé via `scripts/install_onnxruntime.py`, pas comme
  dépendance Python directe du package.
