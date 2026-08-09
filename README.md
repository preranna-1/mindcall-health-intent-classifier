# MindCall Health Intent Classifier

Machine learning pipeline that classifies free-text patient queries into 20 fine-grained health-tracking intents (e.g., `get_sleep_data`, `get_heart_rate_data`, `get_steps_data`) for a mental-health digital phenotyping assistant. Built and evaluated on the [MindCall dataset](https://huggingface.co/datasets/frshafi/mind_call).

Before a conversational health assistant can reason about a patient's behavior, it first has to figure out *what* the patient is actually asking about. This project tackles that routing step: given an open-ended question like *"I wonder if my inconsistent sleep is affecting my mood"*, the model predicts which underlying data function should be called (`get_sleep_data`), so downstream systems can retrieve the right signal.

## What's in this repo

- A text-classification pipeline comparing three models (Logistic Regression, Linear SVM, Random Forest) on TF-IDF features
- An explainability layer using SHAP (global feature importance) and LIME (per-query local explanations)
- An unsupervised knowledge-discovery analysis using Sentence-BERT embeddings and clustering (K-Means, Agglomerative, DBSCAN) to check whether the query intents are naturally separable in semantic space

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.786 | 0.799 | 0.786 | 0.788 |
| **Linear SVM** | **0.799** | **0.801** | **0.799** | **0.798** |
| Random Forest | 0.773 | 0.781 | 0.773 | 0.774 |

The Linear SVM performed best, correctly classifying ~80% of patient queries across 20 classes. Strongest classes: `get_sleep_data`, `get_body_temp_data`, `get_heart_rate_data` (F1 ≥ 0.91). Weakest: `get_calories_burned_data`, `get_speed_data`, `get_exercise_session_data` — these get confused with each other because patients often describe a single activity using overlapping vocabulary (e.g., a workout query mentioning both "calories" and "distance").

Clustering the same queries with Sentence-BERT embeddings (independent of the labels) showed only moderate agreement with the true classes (Adjusted Rand Index ≈ 0.37), confirming from a second angle that these 20 intents sit in a semantically continuous space rather than cleanly separated categories.

## Explainability

- **SHAP** shows the model relies on clinically intuitive keywords (e.g., "sleep," "pressure," "blood," "heart") rather than spurious correlations.
- **LIME** shows the model is confident and accurate on explicit queries (e.g., "check my blood pressure" → correctly driven by the tokens "pressure" and "blood") but noticeably less reliable on implicit or metaphorical phrasing (e.g., "my heart is a lullaby during knitting"), where token attributions are weak and non-specific.

## Dataset

[MindCall](https://huggingface.co/datasets/frshafi/mind_call) — 3,824 train / 510 validation / 765 test samples, 20 balanced intent classes (178–208 samples each), with fields for the patient query, query type (implicit/explicit/behavioral), and the target function label.

## Pipeline

1. **Preprocessing** — lowercase, strip URLs/punctuation, collapse whitespace; median-impute missing `num_days`
2. **Feature extraction** — TF-IDF (5,000 max features) for classification; Sentence-BERT (`all-MiniLM-L6-v2`) embeddings for clustering
3. **Classification** — Logistic Regression, Linear SVM, Random Forest
4. **Explainability** — SHAP summary plots, LIME local explanations
5. **Knowledge discovery** — K-Means, Agglomerative Clustering, DBSCAN on embeddings; evaluated with silhouette score and Adjusted Rand Index

## What this project does *not* do

This model classifies query **intent**, not mental-health **risk or diagnosis**. It doesn't predict depression, anxiety, or any clinical outcome — it only determines which data-retrieval function a patient's question implies. It's designed as an upstream routing layer that could feed a downstream, LLM-based reasoning system.

## Tech stack

`scikit-learn` · `sentence-transformers` · `SHAP` · `LIME` · `pandas` · `matplotlib`

## Requirements

```bash
pip install scikit-learn sentence-transformers shap lime datasets pandas matplotlib
```

## License

Add your preferred license here (e.g., MIT).
