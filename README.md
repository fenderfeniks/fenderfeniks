# Максим Новиков

ML Engineer — строю production-grade системы машинного обучения: от табличных моделей и кредитного скоринга до LLM fine-tuning и машинного перевода для низкоресурсных языков.

Специализация: полный цикл ML-проекта — подготовка данных, обучение, трекинг экспериментов, serving, оркестрация переобучения, мониторинг. Основной стек: Python, CatBoost / LightGBM / XGBoost, PyTorch, Hydra, MLflow, Airflow, FastAPI, Docker.

---

## Проекты

Проекты расположены в порядке от ранних к более зрелым — каждый следующий строится на опыте предыдущего.

---

### Ранние проекты — NLP и CV с базовым деплоем

Три проекта, в которых сформировался базовый пайплайн: модель → FastAPI / Gradio → Docker. Без полной MLOps-обёртки, но с реальными архитектурными и исследовательскими решениями.

---

**[fake-news-detection-ml-system](https://github.com/fenderfeniks/fake-news-detection-ml-system)**

Классификация фейковых новостей через ансамбль рекуррентных архитектур. Полный ML-пайплайн: предобработка текста (очистка URL, стоп-слова, Word2Vec-эмбеддинги) → три модели (RNN / LSTM / GRU) → взвешенный ансамбль → FastAPI → Docker.

Ключевые решения: все три архитектуры обучены независимо с подбором гиперпараметров (hidden dim, num_layers, dropout, lr) через Optuna по ROC AUC; веса ансамбля также оптимизированы Optuna — не фиксированное усреднение, а подбор оптимальной комбинации. Каждая модель возвращает вероятность через API, итоговое решение принимается ансамблем. Метрики: Accuracy, F1, ROC AUC.

---

**[Background-Removal-and-Replacement](https://github.com/fenderfeniks/Background-Removal-and-Replacement)**

CV-проект по удалению и замене фона на изображениях с benchmarking'ом двух сегментационных моделей.

Исследование: сравнение **BiRefNet** и **RMBG** на датасете **DUTS-TE** по метрикам IoU, Dice, MAD. BiRefNet показал лучшее качество (IoU `0.9337`, Dice `0.9651`, MAD `0.0098`), RMBG — IoU `0.9288`. Дополнительно построен и оценён взвешенный ансамбль `0.85 · BiRefNet + 0.15 · RMBG`, который улучшил overlap-метрики относительно бейзлайна. Продукт упакован в интерактивный **Gradio**-интерфейс с тремя режимами замены фона: сплошной цвет, прозрачный PNG, произвольное изображение.

---

**[EMNIST-Handwritten-Character-Recognition](https://github.com/fenderfeniks/EMNIST-Handwritten-Character-Recognition)**

Распознавание рукописных символов (47 классов: цифры + буквы) на датасете EMNIST Balanced (112 800 обучающих, 18 800 тестовых изображений), деплой через FastAPI с интерактивным фронтендом.

Последовательное сравнение архитектур: Logistic Regression (~67%) → MLP с BatchNorm и Dropout (~86%) → CNN (~88%+). Особое внимание уделено предобработке входа при инференсе: данные EMNIST приходят транспонированными, поэтому пайплайн включает транспонирование, морфологическую дилатацию (OpenCV) для утолщения линий, Bilateral Filter, центрирование по center_of_mass и стандартизацию. Финальная модель обучена в Google Colab (T4, 50–100 эпох, ReduceLROnPlateau scheduler) и задеплоена в Docker.

---

### Kaggle — соревновательный ML

---

**[Kaggle-House-Prices-Top5pct-result](https://github.com/fenderfeniks/Kaggle-House-Prices-Top5pct-result)**

House Prices: Advanced Regression Techniques — **Top 5% лидерборда**.

Feature engineering на структурированных данных о недвижимости: взаимодействия признаков, кодирование категорий, обработка выбросов, работа с пропусками. Ансамблирование градиентных бустингов (CatBoost / LightGBM / XGBoost) со стекингом. Основная метрика — RMSE логарифма цены.

---

**[Kaggle_Contradictory_My_Dear_Watson](https://github.com/fenderfeniks/Kaggle_Contradictory_My_Dear_Watson)**

Natural Language Inference на 15 языках (трёхклассовая классификация: Entailment / Neutral / Contradiction). Финальный результат: Test Accuracy **0.773**, Test F1 **0.773**.

Серьёзное исследование: smoke-тест четырёх предобученных NLI-архитектур (DeBERTa v3, XLM-RoBERTa, ModernBERT) показал доминирование DeBERTa — Disentangled Self-Attention лучше всего работает для cross-attention между посылкой и гипотезой. Ключевое архитектурное решение — кастомный `DynamicTextCollator`: наивная конкатенация текстов давала ~33% accuracy (случайное угадывание); коллатор корректно подаёт пару как `text` + `text_pair`, что позволяет токенизатору вставить `[SEP]` и сгенерировать `token_type_ids`. LoRA-настройка через overfitting probe: все комбинации target modules сходились к Acc=1.0; выбраны `query_proj, key_proj, value_proj` (0.24% обучаемых параметров) как минимальная конфигурация с достаточной выразительностью. Вокруг задачи выстроен полный MLOps-стек: MLflow + Model Registry, FastAPI, Airflow DAGs, Docker/K8s, Telegram-бот, Prometheus/Grafana.

---

### Production ML — табличные модели с полным MLOps-циклом

---

**[credit-risk-model](https://github.com/fenderfeniks/credit-risk-model)**

Промышленный пайплайн кредитного скоринга: предсказание вероятности дефолта по агрегированной кредитной истории. ROC-AUC **0.743** на PROD-выборке 420k+ клиентов.

Ключевые решения: SQL-агрегация (DuckDB dev / PySpark prod), кастомный `FeatureEngineer` с макро-индексами и бизнес-флагами, anti-leakage сплиттинг по `client_id`, SHAP-интерпретируемость, тюнинг threshold вместо жёсткой балансировки весов. Полный MLOps: Hydra + MLflow + Optuna + FastAPI + Prometheus + Airflow DAGs (retrain / batch inference / deploy) + Docker.

---

**[sber_autopodpiska](https://github.com/fenderfeniks/sber_autopodpiska)**

Предсказание подписочного поведения клиентов. Построен на той же инфраструктурной базе, что и credit-risk-model, с акцентом на feature engineering под подписочную механику.

---

### NLP с MLOps — развитие от модели к шаблону

---

**[fake-news-detection-ml-system-mlops](https://github.com/fenderfeniks/fake-news-detection-ml-system-mlops)**

MLOps-переработка исходного fake-news проекта на базе задачи классификации email (spam / ham, TREC 2006, 72.3% / 27.7% имбаланс классов). Полный production-пайплайн с заменой архитектуры и добавлением MLOps-обёртки.

Исследование: бенчмарк четырёх BERT-энкодеров под идентичными условиями (3 эпохи, head-only fine-tuning). Победитель — `deepset/bert-base-cased-squad2`: лучший F1 (0.700) и наименьший loss при времени обучения 3.4 мин. Автоматическая балансировка через inverse-frequency weights дала прирост Val F1 с 0.706 до **0.922**. LoRA ablation: overfitting probe на 100 шагах показал, что конфигурация `r=8, α=8, target=[query, value]` (0.17% обучаемых параметров) достаточна — всё сходится к Acc=1.0. Выбор `max_length=256` против 512 обоснован компромиссом recall / скорость (6.6 мин → 3.4 мин). Финальные метрики на ресурсно-ограниченном локальном запуске: Val F1 **0.9655**, Test F1 **0.9354**.

Стек: PyTorch Lightning + PEFT/LoRA + Hydra + MLflow Registry (Staging/Production) + FastAPI + Telegram-бот (aiogram, webhook + polling) + Prometheus/Grafana + Airflow (KubernetesPodOperator) + Kubernetes + GitHub Actions CI.

---

### LLM Fine-Tuning — CPT + SFT + полный MLOps

---

**[machine_translate_rus_abkhaz](https://github.com/fenderfeniks/machine_translate_rus_abkhaz)**

Машинный перевод с русского на абхазский язык — один из наиболее морфологически сложных и низкоресурсных языков Кавказа. Полный цикл: Continual Pre-Training (CPT) на монолингвальном корпусе → Supervised Fine-Tuning (SFT) на 147k параллельных парах → деплой.

Архитектурные решения: скрининг моделей через smoke-tests (Qwen3-4B vs phi-4 — перплексия phi-4 ~1757 на абхазском алфавите против ~43 у Qwen), анализ P99 длин токенов для выбора `packing_chunk_size=512`, MinHash LSH дедупликация, LoRA-aware чекпоинтинг (сохраняет только дельта-веса), маскировка промпта через `prompt_len`, `GenerationEvaluationCallback` с BLEU и таблицами генераций в MLflow на каждом шаге валидации.

Инфраструктура: Hydra + PyTorch Lightning + MLflow Registry (Staging/Production) + PEFT/LoRA + BitsAndBytes 4-bit + Flash Attention 2 + Airflow DAGs + Helm chart (K8s) + FastAPI с SSE стримингом + Telegram-бот + Streamlit демо + Prometheus/Grafana + GitHub Actions CI.

---

### Шаблоны — переиспользуемая инфраструктура

---

**[nlptemplate_decoder_rag](https://github.com/fenderfeniks/nlptemplate_decoder_rag)**

Production-ready шаблон для задач генерации текста (Decoder / LLM) и семантического поиска (RAG). Цель — от датасета до работающего API за минимальное количество шагов, без написания инфраструктурного кода с нуля.

Два полных пайплайна обучения: SFT/CPT с LoRA/QLoRA (PyTorch Lightning) и дообучение RAG-энкодера через contrastive learning (MNRL / Triplet loss). Три микросервиса: API Gateway (RAGOrchestrator + PromptManager), RAG API (FAISS Flat/HNSW), LLM API (vLLM, OpenAI-compatible). Готовые конфиги архитектур: Qwen2.5-1.5B, Qwen3-4B-Instruct, Phi-4-mini, bge-m3. 9 Airflow DAG-ов (retrain, promote, quality control, analytics — для обоих пайплайнов). Helm chart, Streamlit демо, полный тест-сьют с моками. Вся конфигурация через Hydra — смена модели, квантизации, индекса или лосса через CLI без правки кода.

---

## Стек

```
Модели         CatBoost · LightGBM · XGBoost · PyTorch · Transformers · PEFT/LoRA · pytorch_lightning
Конфигурация   Hydra · OmegaConf
Трекинг        MLflow
Данные         DuckDB · PySpark · Pandas · DVC · datasets
Serving        FastAPI · uvicorn · vLLM · aiogram · SlowAPI
Мониторинг     Prometheus · Grafana
Оркестрация    Airflow · Kubernetes · Helm
Инфраструктура Docker · GitHub Actions · uv · ruff
```

---

## Образование

- Бакалавр, менеджмент — Уральский федеральный университет, 2019.
- Повышение квалификации — Skillbox, Data Science / ML Engineer, 800 академических часов. В рамках курса реализовал 20 практических работ: EDA, классические ML-модели, CV (YOLO/SSD/R-CNN), NLP, GAN/VAE, RL (DQN).

### Самостоятельное обучение

1. **Прикладная математика для машинного обучения** — Дианкин И.Д., Пензар Д.Д. (ф-т биоинженерии и биоинформатики МГУ). [Открытый курс](https://teach-in.ru/course/applied-mathematics-for-machine-learning)
2. **Прикладное машинное обучение / ML-basic course spring 2024** — Нейчев Р., Гончаренко В. (МФТИ). [YouTube](https://www.youtube.com/watch?v=MOSNeCYa_bs&list=PLJR10EXrBaAtNQWNssJrFtIF7d4sb9P50)
3. **NLP & RL** — Карпачев Н., Нейчев Р., Лунева Н. (МФТИ). [YouTube](https://www.youtube.com/watch?v=r9cLXcOczTI&list=PLJR10EXrBaAvvUfbs_ZAr0biScOl4Udcb&index=15)

---

mallienotxc@gmail.com · Telegram: [@fenderfeniks](https://t.me/fenderfeniks)
