# 🏥 Riesgo de Reingreso Hospitalario

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![sklearn](https://img.shields.io/badge/scikit--learn-ML-green?logo=scikit-learn)
![Status](https://img.shields.io/badge/Estado-Completo-brightgreen)

**Inteligencia Artificial 1 — Grupo C2 2026-1**  
**Universidad Industrial de Santander**

</div>

---

## 👥 Equipo — IA Médica

| Integrante | Rol |
|-----------|-----|
| Cesar Daniel Ávila Barbosa | Modelado supervisado, no supervisado y DL |
| Tatiana Lineros Sánchez | EDA, preprocesamiento y análisis clínico |

---

## 📋 Descripción del problema

El **reingreso hospitalario dentro de los 30 días** posteriores al alta es uno de los principales indicadores de calidad asistencial. Un paciente readmitido puede reflejar tratamiento inefectivo, seguimiento insuficiente o complicaciones clínicas no detectadas.

Este proyecto desarrolla modelos de Machine Learning para **predecir qué pacientes tienen mayor riesgo de reingresar**, con foco en minimizar los Falsos Negativos (pacientes de riesgo no detectados), ya que clínicamente son más críticos que las falsas alarmas.

### Desafíos técnicos
- **Desbalance de clases:** 74.2% reingreso vs 25.8% no reingreso → accuracy no es métrica válida
- **Baja separabilidad lineal:** PCA reveló que PC1+PC2 explican solo el 10.1% de varianza
- **Importancia del Recall:** un FN implica un paciente que se va sin plan de seguimiento

---

## 📂 Estructura del repositorio

```
ia1-iaMedica-riesgo_reingreso_hospitalario/
│
├── exploratory_data_analysis.ipynb     # Corte 1 — EDA completo
├── supervised_learning.ipynb           # Corte 2 — Modelos supervisados
├── unsupervised_learning.ipynb         # Corte 3 — No supervisado + Deep Learning
├── data/
│   └── README_data.md                  # Instrucciones para obtener el dataset
└── README.md
```

---

## 📊 Datos

| Campo | Detalle |
|-------|---------|
| **Dataset** | Hospital Readmission Risk Dataset |
| **Fuente** | [Kaggle](https://www.kaggle.com/) — buscar `Hospital Readmission Risk Dataset` |
| **Registros** | 18,000 pacientes |
| **Variables** | 25 (clínicas, demográficas e historial hospitalario) |
| **Variable objetivo** | `Reingreso_30_Dias` (1 = reingreso en 30 días, 0 = no reingreso) |
| **Desbalance** | 74.2% positivo / 25.8% negativo |

### Variables principales

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `Edad` | Numérica | Edad del paciente |
| `Puntaje_Severidad` | Numérica | Severidad clínica al ingreso (1–9) |
| `Numero_Medicamentos` | Numérica | Medicamentos prescritos al alta |
| `Hospitalizaciones_6M` | Numérica | Hospitalizaciones en últimos 6 meses |
| `Nivel_HbA1c` | Numérica | Hemoglobina glicosilada |
| `Nivel_Creatinina` | Numérica | Indicador de función renal |
| `Genero` | Categórica | M / F |
| `Diagnostico_Principal` | Categórica | Diagnóstico al ingreso |
| `Reingreso_30_Dias` | Binaria | **Target** — 0/1 |

### Cómo obtener el dataset

1. Crear cuenta en [Kaggle](https://www.kaggle.com/) (gratuita)
2. Buscar `Hospital Readmission Risk Dataset`
3. Descargar el archivo `.csv`
4. Renombrarlo a `hospital_readmission.csv` y colocarlo en `data/`
5. El preprocesamiento completo está en `exploratory_data_analysis.ipynb`

---

## 🧪 Notebooks

### 1. `exploratory_data_analysis.ipynb` — Corte 1: EDA
> **62 celdas** — Análisis exploratorio completo

- Carga del dataset y renombrado de variables
- Detección y corrección de outliers (creatinina negativa)
- Distribuciones de variables numéricas y categóricas
- Análisis bivariado: tasa de reingreso por variable
- Matriz de correlaciones con el target
- Conclusiones y próximos pasos

### 2. `supervised_learning.ipynb` — Corte 2: Modelos Supervisados
> **34 celdas** — Pipeline supervisado completo

- Preprocesamiento: One-Hot Encoding + StandardScaler
- División Train/Test 80/20 estratificada
- Validación cruzada 5-Fold estratificada
- Modelos evaluados: GaussianNB, DecisionTree, RandomForest, SVM
- GridSearchCV para tuning de hiperparámetros
- Curvas ROC, matrices de confusión, análisis de FN por subgrupo
- Feature Importance (RandomForest)
- Análisis del umbral de decisión
- Score Clínico ponderado: 40% AUROC + 40% Recall + 20% F1

### 3. `unsupervised_learning.ipynb` — Corte 3: No Supervisado + Deep Learning
> **32 celdas** — Técnicas avanzadas y modelo final

- **MLP (Red Neuronal):** arquitectura 64→32 ReLU, GridSearchCV, análisis de overfitting
- **PCA:** scree plot, varianza acumulada, scatter 2D coloreado por clase
- **t-SNE:** 3 perplexidades (10/30/50) para visualización no lineal
- **K-Means:** método del Silhouette, k=4, perfiles clínicos por cluster
- **DBSCAN:** eps calibrado automáticamente con k-distancia
- Interpretación clínica profunda de los clusters
- Tabla comparativa final: todos los modelos vs MLP
- **Simulación de predicción clínica** en tiempo real

---

## 📈 Resultados

### Modelos supervisados — Test set (3,600 muestras)

| Modelo | AUROC | Recall | FN | FP | Score Clínico |
|--------|-------|--------|----|----|---------------|
| **GaussianNB** ★ | 0.5648 | 1.0000 | **0** | 930 | **0.7963** |
| DecisionTree | 0.5401 | 1.0000 | 0 | 930 | 0.7864 |
| MLP — Red Neuronal | 0.5556 | 0.9375 | 167 | 839 | 0.7638 |
| RandomForest | 0.5759 | 0.5899 | 1,095 | 457 | 0.6003 |
| SVM | 0.5663 | 0.5232 | 1,273 | 407 | 0.5607 |

### Hallazgos del análisis no supervisado

| Técnica | Hallazgo |
|---------|---------|
| PCA | PC1+PC2 = 10.1% varianza → clases solapadas linealmente → explica AUROC ~0.56 |
| t-SNE | Subpoblaciones clínicas visibles con estructura no lineal |
| K-Means | 4 perfiles clínicos con distintas tasas de reingreso |
| DBSCAN | Pacientes atípicos concentran mayor tasa de reingreso |

> **Modelo recomendado:** GaussianNB con Score Clínico 0.7963 y **0 Falsos Negativos** — clínicamente el más seguro.

---

## 🎥 Video resumen

▶️ [Ver video del proyecto](https://drive.google.com/file/d/1DVMITq6iCKRQi-jkk5H4FisTpUrhZehZ/view?usp=sharing)

---

## ⚙️ Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn missingno
```

> El proyecto fue desarrollado en **Google Colab**. Para correrlo localmente, reemplazar la celda de montaje de Drive por la ruta local del dataset.

---

## 🏷️ Topics

`uis-ia1` · `uis-ia1-c2`
