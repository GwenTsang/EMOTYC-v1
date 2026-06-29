# Architecture v1

La v1 couvre l'inference, l'evaluation, la comparaison de resultats et
l'entrainement lineaire.
Les modules sont organises pour eviter la duplication entre les commandes CLI.

## Modules partages

- `emotyc.formatting` applique les templates `raw` et `bca`, avec contexte adjacent optionnel pour `bca`.
- `emotyc.encoders` expose le protocole `Encoder` et `OnnxBackboneEncoder`.
- `emotyc.heads` charge et valide la tete de classification v1.
- `emotyc.inference` execute l'inference batch et applique le seuil global.
- `emotyc.predictors` assemble un bundle modele en predictor utilisable par les CLI.
- `emotyc.linear` entraine et charge des tetes lineaires TF-IDF ou ONNX.
- `emotyc.metrics` calcule les metriques multi-labels avec NumPy.
- `emotyc.io` lit/ecrit le JSON et exporte les predictions XLSX.
- `emotyc.compare` compare deux fichiers `metrics.json`.

## Separation encoder / tete / predictor

`OnnxBackboneEncoder` transforme les textes en features. La tete transforme les
features en logits. `run_inference` transforme logits, probabilites et seuil en
predictions. `Predictor` assemble ces pieces pour les bundles EMOTYC v1.

L'encodeur ONNX tokenise avec `add_special_tokens=False`, comme le pipeline
EMOTYC/BCA de reference. Les textes sont tries par longueur tokenisee pour
former les batches GPU, puis les features sont restaurees dans l'ordre initial.

`train_linear.py` reutilise cette separation pour entrainer une tete `LinearSVC`
par label sur des features TF-IDF ou des embeddings ONNX. `scikit-learn` reste
une dependance optionnelle limitee a l'extra `train`.
