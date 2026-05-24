# 🏥 Riesgo de Reingreso Hospitalario

**Equipo:** IA Médica  
**Integrantes:** Cesar Daniel Ávila Barbosa · Tatiana Lineros Sánchez  
**Curso:** Inteligencia Artificial 1 — Grupo C2 2026-1 — UIS

## Problema
Predecir si un paciente reingresará al hospital dentro de los 30 días posteriores al alta, usando variables clínicas y demográficas.

## Datos
- Dataset: Hospital Readmission Risk Dataset (Kaggle)
- 18,000 pacientes | 25 variables | Target: Reingreso_30_Dias
- Ver instrucciones de descarga en data/README_data.md

## Notebooks
- exploratory_data_analysis.ipynb — Corte 1: EDA completo
- supervised_learning.ipynb — Corte 2: GaussianNB, DT, RF, SVM, MLP
- unsupervised_learning.ipynb — Corte 3: PCA, t-SNE, K-Means, DBSCAN

## Video resumen
[Ver video del proyecto](https://drive.google.com/file/d/1DVMITq6iCKRQi-jkk5H4FisTpUrhZehZ/view?usp=sharing)

## Resultados
| Modelo | Recall | FN | Score Clínico |
|--------|--------|----|---------------|
| GaussianNB | 1.0000 | 0 | 0.7963 ★ |
| MLP (C3) | 0.9375 | 167 | 0.7638 |
| RandomForest | 0.5899 | 1,095 | 0.6003 |
