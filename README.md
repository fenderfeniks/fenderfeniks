# Максим Новиков

NLP / LLM Engineer — специализируюсь на дообучении языковых моделей и построении production-ready NLP-систем: от классификации и NLI до LLM fine-tuning, машинного перевода и RAG-пайплайнов.

Основной фокус — **полный цикл NLP-проекта**: подготовка текстовых данных (очистка, дедупликация, чанкование), дообучение (CPT / SFT / LoRA / QLoRA), оценка качества (BLEU, ROUGE, NLI-judge, LLM-as-a-Judge), serving (FastAPI, vLLM, SSE-стриминг), оркестрация переобучения и мониторинг в продакшне. Параллельно — опыт в tabular ML (кредитный скоринг, CatBoost/LightGBM/XGBoost) и CV.

Стек: Python · PyTorch · Transformers · PEFT/LoRA · PyTorch Lightning · Hydra · MLflow · Airflow · FastAPI · vLLM · Docker · Kubernetes

---

## Проекты

Расположены тематически — сначала NLP/LLM (основная специализация), затем tabular ML и ранние работы.

---

### LLM Fine-Tuning и RAG — основная специализация

---

**[nlptemplate_decoder_rag](https://github.com/fenderfeniks/nlptemplate_decoder_rag)**

Production-ready шаблон для задач генерации текста (Decoder / LLM) и семантического поиска (RAG). Главная идея: **смоук-тест любой модели на своих данных без обучения** — если zero-shot результаты устраивают, сервис готов к деплою буквально за день.

Два полных пайплайна дообучения. Для декодера: CPT (sequence packing — конкатенация потока текстов и нарезка на блоки фиксированного размера в токенах) и SFT (`prompt + separator + target` с маскировкой промпта в loss). Для RAG: contrastive learning энкодера через MNRL / Triplet loss, чанкование документов с перекрытием, индексация FAISS (Flat / HNSW) или Qdrant. Оценка генерации прямо во время SFT через `GenerationEvaluationCallback`: BLEU, ROUGE + опционально LLM-as-a-Judge (OpenRouter) или NLI-judge (RoBERTa) — подключаются через конфиг, инициализируются лениво. Транспортировка модели через манифест: `promote` (LoRA из mlruns → storage, сравнение по val_loss), `merge_lora` (LoRA + base → merged model); три бэкенда хранилища — local, S3, HF Hub.

Три микросервиса: API Gateway (RAGOrchestrator + PromptManager с шаблонами из YAML), RAG API (FAISS-поиск), Decoder API (vLLM, OpenAI-compatible). 10 Airflow DAG-ов (retrain, promote, quality control, analytics — для обоих пайплайнов). Helm chart, Streamlit-демо, Telegram-бот, тест-сьют ~80% покрытие. Вся конфигурация через Hydra — смена модели, квантизации, лосса или индекса через CLI.

---

**[machine_translate_rus_abkhaz](https://github.com/fenderfeniks/machine_translate_rus_abkhaz)**

Машинный перевод с русского на абхазский язык — один из наиболее морфологически сложных и низкоресурсных языков Кавказа. Полный цикл: Continual Pre-Training (CPT) на монолингвальном корпусе → Supervised Fine-Tuning (SFT) на 147k параллельных парах → деплой.

Ключевые NLP-решения: скрининг базовых моделей через smoke-tests по перплексии (Qwen3-4B ~43 vs phi-4 ~1757 на абхазском алфавите), анализ P99 длин токенов для выбора `packing_chunk_size=512`, MinHash LSH дедупликация корпуса, LoRA-aware чекпоинтинг (сохраняет только дельта-веса), маскировка промпта через `prompt_len`, `GenerationEvaluationCallback` с BLEU и таблицами генераций в MLflow на каждой валидации.

Инфраструктура: Hydra + PyTorch Lightning + MLflow Registry (Staging/Production) + PEFT/LoRA + BitsAndBytes 4-bit + Flash Attention 2 + Airflow DAGs + Helm (K8s) + FastAPI с SSE стримингом + Telegram-бот + Streamlit-демо + Prometheus/Grafana + GitHub Actions CI.

---

**[Kaggle_Contradictory_My_Dear_Watson](https://github.com/fenderfeniks/Kaggle_Contradictory_My_Dear_Watson)**

Natural Language Inference на 15 языках (трёхклассовая классификация: Entailment / Neutral / Contradiction). Финальный результат: Test Accuracy **0.773**, Test F1 **0.773**.

Исследование архитектур: smoke-тест четырёх NLI-моделей (DeBERTa v3, XLM-RoBERTa, ModernBERT) показал доминирование DeBERTa — Disentangled Self-Attention лучше всего работает для cross-attention между посылкой и гипотезой. Ключевое инженерное решение — кастомный `DynamicTextCollator`: наивная конкатенация текстов давала ~33% accuracy (случайное угадывание); коллатор корректно подаёт пару как `text` + `text_pair`, что позволяет токенизатору вставить `[SEP]` и сгенерировать `token_type_ids`. LoRA-настройка через overfitting probe: выбраны `query_proj, key_proj, value_proj` (0.24% обучаемых параметров) как минимальная конфигурация с достаточной выразительностью.

Вокруг задачи выстроен полный MLOps-стек: MLflow + Model Registry, FastAPI, Airflow DAGs, Docker/K8s, Telegram-бот, Prometheus/Grafana.

---

**[fake-news-detection-ml-system-mlops](https://github.com/fenderfeniks/fake-news-detection-ml-system-mlops)**

MLOps-переработка классификатора текста: задача spam/ham (TREC 2006, 72.3% / 27.7% имбаланс). Фокус — методичное исследование пространства решений и production-обёртка.

Бенчмарк четырёх BERT-энкодеров под идентичными условиями (3 эпохи, head-only fine-tuning). Победитель — `deepset/bert-base-cased-squad2`: лучший F1 (0.700) при времени обучения 3.4 мин. Автоматическая балансировка через inverse-frequency weights — прирост Val F1 с 0.706 до **0.922**. LoRA ablation: overfitting probe на 100 шагах, конфигурация `r=8, α=8, target=[query, value]` (0.17% параметров) оказалась достаточной. Финальные метрики: Val F1 **0.9655**, Test F1 **0.9354**.

Стек: PyTorch Lightning + PEFT/LoRA + Hydra + MLflow Registry + FastAPI + aiogram + Prometheus/Grafana + Airflow (KubernetesPodOperator) + Kubernetes + GitHub Actions CI.

---

**[fake-news-detection-ml-system](https://github.com/fenderfeniks/fake-news-detection-ml-system)**

Ранний NLP-проект: классификация фейковых новостей через ансамбль рекуррентных архитектур. Предобработка текста (очистка, стоп-слова, Word2Vec) → три модели (RNN / LSTM / GRU) → взвешенный ансамбль → FastAPI → Docker.

Примечательное решение: веса ансамбля подобраны Optuna по ROC AUC — не фиксированное усреднение, а оптимальная комбинация из трёх независимо обученных моделей.

---

### Production ML — tabular-модели с полным MLOps-циклом

---

**[credit-risk-model](https://github.com/fenderfeniks/credit-risk-model)**

Промышленный пайплайн кредитного скоринга: предсказание вероятности дефолта по агрегированной кредитной истории. ROC-AUC **0.743** на PROD-выборке 420k+ клиентов.

Ключевые решения: SQL-агрегация (DuckDB dev / PySpark prod), кастомный `FeatureEngineer` с макро-индексами и бизнес-флагами, anti-leakage сплиттинг по `client_id`, SHAP-интерпретируемость, тюнинг threshold вместо жёсткой балансировки весов. Полный MLOps: Hydra + MLflow + Optuna + FastAPI + Prometheus + Airflow DAGs (retrain / batch inference / deploy) + Docker.

---

**[sber_autopodpiska](https://github.com/fenderfeniks/sber_autopodpiska)**

Предсказание подписочного поведения клиентов. Построен на той же инфраструктурной базе, что и credit-risk-model, с акцентом на feature engineering под подписочную механику.

---

**[Kaggle-House-Prices-Top5pct-result](https://github.com/fenderfeniks/Kaggle-House-Prices-Top5pct-result)**

House Prices: Advanced Regression Techniques — **Top 5% лидерборда**. Feature engineering на структурированных данных, ансамблирование градиентных бустингов (CatBoost / LightGBM / XGBoost) со стекингом.

---

### CV-проекты

---

**[Background-Removal-and-Replacement](https://github.com/fenderfeniks/Background-Removal-and-Replacement)**

Сравнение BiRefNet и RMBG на датасете DUTS-TE (IoU, Dice, MAD), взвешенный ансамбль `0.85 · BiRefNet + 0.15 · RMBG`, Gradio-интерфейс с тремя режимами замены фона.

---

**[EMNIST-Handwritten-Character-Recognition](https://github.com/fenderfeniks/EMNIST-Handwritten-Character-Recognition)**

Распознавание рукописных символов (47 классов, EMNIST Balanced). Последовательное сравнение Logistic Regression → MLP → CNN. Нетривиальный инференс-пайплайн: транспонирование, морфологическая дилатация, Bilateral Filter, центрирование по `center_of_mass`.

---

## Стек

```
NLP / LLM      Transformers · PEFT/LoRA · PyTorch Lightning · vLLM · BitsAndBytes · Flash Attention
               datasets · sacrebleu · rouge-score · evaluate · FAISS · Qdrant
Tabular ML     CatBoost · LightGBM · XGBoost · Optuna · SHAP
Конфигурация   Hydra · OmegaConf
Трекинг        MLflow · MLflow Registry
Данные         DuckDB · PySpark · Pandas · DVC · datasketch (MinHash)
Serving        FastAPI · uvicorn · vLLM · aiogram · slowapi · SSE
Мониторинг     Prometheus · Grafana
Оркестрация    Airflow · Kubernetes · Helm
Инфраструктура Docker · GitHub Actions · uv · ruff · pytest
```

---

## Образование

- Бакалавр, Математическое моделирование / Менеджмент в IT — Уральский федеральный университет, 2019.
- Повышение квалификации — Skillbox, Data Science & Machine Learning Engineer, 800 академических часов (EDA, классические ML-модели, CV, NLP, GAN/VAE, RL).

### Самостоятельное обучение

1. **Прикладная математика для машинного обучения** — Дианкин И.Д., Пензар Д.Д. (МГУ). [Открытый курс](https://teach-in.ru/course/applied-mathematics-for-machine-learning)
2. **ML-basic course spring 2024** — Нейчев Р., Гончаренко В. (МФТИ). [YouTube](https://www.youtube.com/watch?v=MOSNeCYa_bs&list=PLJR10EXrBaAtNQWNssJrFtIF7d4sb9P50)
3. **NLP & RL** — Карпачев Н., Нейчев Р., Лунева Н. (МФТИ). [YouTube](https://www.youtube.com/watch?v=r9cLXcOczTI&list=PLJR10EXrBaAvvUfbs_ZAr0biScOl4Udcb&index=15)

---

mallienotxc@gmail.com · Telegram: [@fenderfeniks](https://t.me/fenderfeniks)
