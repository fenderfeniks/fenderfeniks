# Максим Новиков

ML Engineer — строю production-grade системы машинного обучения: от табличных моделей и кредитного скоринга до LLM fine-tuning и машинного перевода для низкоресурсных языков.

Специализация: полный цикл ML-проекта — подготовка данных, обучение, трекинг экспериментов, serving, оркестрация переобучения, мониторинг. Основной стек: Python, CatBoost / LightGBM / XGBoost, PyTorch, Hydra, MLflow, Airflow, FastAPI, Docker.

---

## Проекты

Проекты расположены в порядке от ранних к более зрелым — каждый следующий строится на опыте предыдущего.

### Ранние проекты — NLP и CV с базовым деплоем

Три проекта где формировался базовый пайплайн: модель → FastAPI → Docker. Без MLOps-обёртки, но с реальными архитектурными решениями.

| Репозиторий | Задача | Стек |
|---|---|---|
| [fake-news-detection-ml-system](https://github.com/fenderfeniks/fake-news-detection-ml-system) | Классификация фейков: RNN / LSTM / GRU + Word2Vec + ансамбль с Optuna | PyTorch, FastAPI, Docker |
| [Background-Removal-and-Replacement](https://github.com/fenderfeniks/Background-Removal-and-Replacement) | Удаление и замена фона на изображениях | CV, сегментация |
| [EMNIST-Handwritten-Character-Recognition](https://github.com/fenderfeniks/EMNIST-Handwritten-Character-Recognition) | Распознавание рукописных символов, деплой в Docker | PyTorch, FastAPI, Docker |

---

### Kaggle — соревновательный ML

| Репозиторий | Соревнование | Результат |
|---|---|---|
| [Kaggle-House-Prices-Top5pct-result](https://github.com/fenderfeniks/Kaggle-House-Prices-Top5pct-result) | House Prices: Advanced Regression Techniques | Top 5% лидерборда |
| [Kaggle_Contradictory_My_Dear_Watson](https://github.com/fenderfeniks/Kaggle_Contradictory_My_Dear_Watson) | Contradictory, My Dear Watson (NLI) | Многоязычная классификация + MLflow |

---

### Production ML — табличные модели с полным MLOps-циклом

**[credit-risk-model](https://github.com/fenderfeniks/credit-risk-model)**

Промышленный пайплайн кредитного скоринга: предсказание вероятности дефолта по агрегированной кредитной истории. ROC-AUC 0.743 на PROD-выборке 420k+ клиентов.

Ключевые решения: SQL-агрегация (DuckDB dev / PySpark prod), кастомный FeatureEngineer с макро-индексами и бизнес-флагами, anti-leakage сплиттинг по client_id, SHAP-интерпретируемость, тюнинг threshold вместо жёсткой балансировки весов. Полный MLOps: Hydra + MLflow + Optuna + FastAPI + Prometheus + Airflow DAGs (retrain / batch inference / deploy) + Docker.

**[sber_autopodpiska](https://github.com/fenderfeniks/sber_autopodpiska)**

Предсказание подписочного поведения клиентов. Построен на той же инфраструктурной базе что и credit-risk-model, с акцентом на feature engineering под подписочную механику.

---

### NLP с MLOps — развитие от модели к шаблону

**[fake-news-detection-ml-system-mlops](https://github.com/fenderfeniks/fake-news-detection-ml-system-mlops)**

Переработка исходного fake-news проекта: заменены архитектуры на более сильные модели, добавлен LoRA fine-tuning, построена MLOps-обёртка. Демонстрирует эволюцию от учебного решения к production-пайплайну на той же задаче.

**[Kaggle_Contradictory_My_Dear_Watson](https://github.com/fenderfeniks/Kaggle_Contradictory_My_Dear_Watson)**

Многоязычная NLI-классификация (Natural Language Inference) на 15 языках. MLflow-трекинг, сравнение архитектур.

---

### LLM Fine-Tuning — CPT + SFT + полный MLOps

**[machine_translate_rus_abkhaz](https://github.com/fenderfeniks/machine_translate_rus_abkhaz)**

Машинный перевод с русского на абхазский язык — один из наиболее морфологически сложных и низкоресурсных языков Кавказа. Полный цикл: Continual Pre-Training (CPT) на монолингвальном корпусе → Supervised Fine-Tuning (SFT) на 147k параллельных парах → деплой.

Архитектурные решения: скрининг моделей через smoke-tests (Qwen3-4B vs phi-4 — перплексия phi-4 ~1757 на абхазском алфавите против ~43 у Qwen), анализ P99 длин токенов для выбора packing_chunk_size=512, MinHash LSH дедупликация, LoRA-aware чекпоинтинг (сохраняет только дельта-веса), маскировка промпта через prompt_len, GenerationEvaluationCallback с BLEU и таблицами генераций в MLflow на каждом шаге валидации.

Инфраструктура: Hydra + PyTorch Lightning + MLflow Registry (Staging/Production алиасы) + PEFT/LoRA + BitsAndBytes 4-bit + Flash Attention 2 + Airflow DAGs + Helm chart (K8s) + FastAPI с SSE стримингом + Telegram бот + Streamlit демо + Prometheus/Grafana + GitHub Actions CI.

---

### В разработке

NLP-шаблон для decoder-like моделей с RAG — индексатор, ретривер, генератор, оркестрация обновления индекса. Шаблонная инфраструктура в приватном репозитории; публичный проект на конкретной задаче появится позже.

---

## Стек

```
Модели         CatBoost · LightGBM · XGBoost · PyTorch · Transformers · PEFT/LoRA · pytorch_lightning
Конфигурация   Hydra · OmegaConf
Трекинг        MLflow 
Данные         DuckDB · PySpark · Pandas · DVC · datasets
Serving        FastAPI · uvicorn · aiogram · SlowApi
Мониторинг     Prometheus · Grafana
Оркестрация    Airflow · Kubernetes · Helm
Инфраструктура Docker · GitHub Actions · uv · ruff
```

---

## Образование

Бакалавр, менеджмент — Уральский федеральный университет, 2019.
Повышение квалификации — Skillbox, Data Science / ML Engineer, 800 академических часов. В рамках курса реализовал 20 практических работ: EDA, классические ML-модели, CV (YOLO/SSD/R-CNN), NLP, GAN/VAE, RL (DQN)


### Самостоятельное обучение

Лекции: 
1. Название курса: Прикладная математика для машинного обучения
   Авторы курса: Дианкин Игорь Дмитриевич, Пензар Дмитрий Дмитриевич
   Образовательное учреждение: Занятия проводятся на факультете биоинженерии и биоинформатики МГУ им. М. В. Ломоносова
   Ссылка на открытый источник: https://teach-in.ru/course/applied-mathematics-for-machine-learning.
2. Название курса: Прикладное машинное обучение/Ml-basic course spring 2024
   Авторы курса: Радослав Нейчев, Владислав Гончаренко
   Образовательное учреждение: Московский физико-технический институт
   Ссылка на открытый источник: https://www.youtube.com/watch?v=MOSNeCYa_bs&list=PLJR10EXrBaAtNQWNssJrFtIF7d4sb9P50
3. Название курса: NLP & RL
   Лектор курса: Николай Карпачев, Радослав Нейчев
   Образовательное учреждение: Московский физико-технический институт
   Ссылка на открытый источник: https://www.youtube.com/watch?v=r9cLXcOczTI&list=PLJR10EXrBaAvvUfbs_ZAr0biScOl4Udcb&index=15

---

mallienotxc@gmail.com · Telegram: [@fenderfeniks](https://t.me/fenderfeniks)
