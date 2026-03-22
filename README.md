# HH.ru Job Market NLP

Exploratory analysis and NLP modelling on 392k Russian job postings from HH.ru, SuperJob, and Rabota.ru.

## What's inside

**EDA**
- Salary distribution (median 96k RUB)
- Top cities and top-20 in-demand skills
- Experience level breakdown

**ML: Experience Level Prediction**
- TF-IDF on skills only → F1 = 0.405
- TF-IDF on skills + title + role → F1 = 0.531
- Binary (Junior vs Senior) → CV ROC-AUC = **0.837**
- SHAP feature importance

**Semantic Resume–Vacancy Matching**
- Multilingual sentence embeddings (MiniLM-L12-v2)
- HR boilerplate preprocessing (+9% similarity improvement)
- Gap analysis: what's missing in resume vs vacancy
- Top-N vacancy search by resume

## Results

| Model | Score |
|---|---|
| TF-IDF skills only | F1 = 0.405 |
| TF-IDF skills + name + role | F1 = 0.531 |
| Junior vs Senior (binary) | AUC = 0.837 |
| Semantic match (raw) | 64.05% |
| Semantic match (preprocessed) | 72.93% |

## Dataset

[HH.ru Vacancies — Mendeley](https://data.mendeley.com/datasets/gkfx465zwk/1) — 575k vacancies, 2022–2023.

Place `db.sqlite` in the project root before running.

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook hh_job_market_nlp.ipynb
```

## Stack

`pandas` `numpy` `scikit-learn` `sentence-transformers` `shap` `matplotlib` `seaborn`
