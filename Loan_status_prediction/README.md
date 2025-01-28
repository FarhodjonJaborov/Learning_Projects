# Прогнозирование Одобрения Кредита

Этот проект посвящен предсказанию того, будет ли кредитная заявка одобрена на основе данных клиента. Задача формулируется как задача классификации, и мы шаг за шагом строим модели машинного обучения. Ниже приведены основные детали и этапы проекта:

---

## 1. Постановка Задачи
Цель проекта — предсказать одобрение кредита, используя данные клиента, путем построения и оценки моделей классификации.

---

## 2. Используемые Библиотеки

Для обработки данных, визуализации и построения моделей используются следующие библиотеки:

```python
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
%matplotlib inline
```

### Назначение Каждой Библиотеки
- **NumPy**: Работа с массивами и выполнение математических вычислений.
- **Pandas**: Используется для манипуляции и анализа данных.
- **Matplotlib и Seaborn**: Предоставляют инструменты для визуализации данных и анализа.
- **ydata-profiling**: Автоматизирует создание подробных отчетов о данных с использованием статистики и визуализаций. Это упрощает понимание и подготовку данных в одной строке кода. Документация: [ydata-profiling](https://docs.profiling.ydata.ai/latest/).
- **SMOTE (Synthetic Minority Oversampling Technique)**: Решает проблему дисбаланса классов, генерируя синтетические образцы для меньшинств, что улучшает производительность моделей классификации на несбалансированных данных.

---

## 3. Профилирование Данных с ydata-profiling

### Установка и Использование
Для установки пакета используйте команду:
```bash
!pip install ydata-profiling
```

Импорт и использование в ноутбуке:
```python
from ydata_profiling import ProfileReport
profile = ProfileReport(df, explorative=True)
profile.to_notebook_iframe()
```
- **explorative=True**: Включает расширенные визуализации и аналитику, делая профилирование данных более полным.

### Обзор Профилирования Данных
Профилирование данных — это процесс анализа наборов данных для понимания их структуры, качества и ключевых характеристик. Это помогает выявлять пропущенные значения, выбросы и распределения.

---

## 4. Обработка Выбросов

Выбросы идентифицируются и удаляются для повышения производительности модели. Используется метод интерквартильного размаха (IQR):

```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```
Это гарантирует, что экстремальные точки данных не искажают анализ или негативно влияют на производительность модели.

---

## 5. Построение Моделей

Мы оцениваем несколько моделей классификации, чтобы определить лучшую. Среди них:
- Логистическая Регрессия
- K-ближайших соседей (KNN)
- Метод Опорных Векторов (SVM)
- Наивный Байес (CategoricalNB и GaussianNB)
- Дерево Решений
- Случайный Лес
- Градиентный Бустинг
- XGBoost

### Выбор и Настройка Модели
Лучшие результаты показала модель Gradient Boosting с точностью **84.06%** на тестовой выборке. Для дальнейшей оптимизации производительности применялась настройка гиперпараметров с использованием `GridSearchCV` и `RandomizedSearchCV`.

---

## 6. Будущая Работа
В будущих проектах мы исследуем метрики регрессии и углубимся в методы оценки моделей регрессии.

---

Спасибо за внимание!


