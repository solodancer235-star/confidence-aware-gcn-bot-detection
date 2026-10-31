# Система обнаружения бот-сетей в Twitter

Прототип системы для обнаружения бот-сетей в социальных сетях с использованием графовых нейронных сетей, семантического анализа и алгоритмов обнаружения сообществ.

## Описание

Система использует комбинацию следующих методов:
- **Графовый анализ**: Анализ структуры связей между пользователями
- **Семантический анализ**: Использование SBERT для обнаружения семантически похожих твитов
- **Обнаружение сообществ**: Алгоритмы Leiden и Spectral Clustering
- **Графовые нейронные сети**: GCN для классификации ботов

## Требования

- Python 3.8 или выше
- pip или conda для установки зависимостей

## Установка

### 1. Установка зависимостей

На Windows используйте:
```bash
py -m pip install -r requirements.txt
```

На Linux/Mac используйте:
```bash
pip install -r requirements.txt
```

### 2. Установка PyTorch Geometric

PyTorch Geometric требует дополнительной настройки. После установки основных зависимостей выполните:

```bash
# Для Windows (CPU)
py -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
py -m pip install torch-geometric

# Для Linux/Mac (CPU)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

**Примечание**: Если у вас установлена GPU версия PyTorch, замените `cpu` на `cu118` (для CUDA 11.8) или соответствующую версию.

## Запуск

### Базовый запуск

```bash
py bot_detection_system.py
```

или

```bash
python bot_detection_system.py
```

### Что происходит при запуске

1. **Генерация данных** (ЭТАП 1):
   - Создается синтетический граф с 500 узлами
   - 400 узлов - "люди" с органическими связями
   - 100 узлов - "боты" в 2 плотных кластерах
   - Каждому узлу присваиваются текстовые атрибуты (твиты)

2. **Семантический анализ** (ЭТАП 2):
   - Загружается модель Sentence-Transformer 'all-MiniLM-L6-v2'
   - Вычисляются эмбеддинги всех твитов
   - Добавляются семантические ребра между узлами с высокой схожестью (>0.85)

3. **Обнаружение сообществ** (ЭТАП 3):
   - Применяется алгоритм Leiden для первичной сегментации
   - Используется Spectral Clustering для поиска структурных аномалий

4. **GNN классификация** (ЭТАП 4):
   - Создается и обучается Graph Convolutional Network
   - Модель предсказывает метки "бот/человек" для каждого узла

5. **Визуализация** (ЭТАП 5):
   - Создается визуализация графа с цветовой кодировкой
   - Сохраняется изображение `bot_detection_graph.png`

## Результаты

После выполнения программа:
- Выводит метрики классификации (точность, количество обнаруженных ботов)
- Сохраняет визуализацию графа в файл `bot_detection_graph.png`
- Показывает цветовую кодировку:
  - 🟢 Зеленый: Правильно предсказанный человек
  - 🔴 Красный: Правильно предсказанный бот
  - 🟠 Оранжевый: Ложная тревога (человек предсказан как бот)
  - 🟡 Желтый: Пропуск (бот предсказан как человек)

## Структура кода

- `generate_synthetic_twitter_graph()` - Генерация синтетических данных
- `add_semantic_edges()` - Добавление семантических ребер через SBERT
- `detect_communities_leiden()` - Обнаружение сообществ алгоритмом Leiden
- `detect_anomalies_spectral()` - Spectral Clustering для поиска аномалий
- `BotDetectionGCN` - Класс GNN модели
- `train_gnn_model()` - Обучение модели
- `visualize_graph()` - Визуализация результатов

## Параметры настройки

В функции `main()` можно изменить следующие параметры:

```python
# Количество людей и ботов
G, labels = generate_synthetic_twitter_graph(n_people=400, n_bots=100, seed=42)

# Порог семантического сходства
G, n_semantic = add_semantic_edges(G, model_sbert, similarity_threshold=0.85)

# Количество кластеров для Spectral Clustering
spectral_clusters = detect_anomalies_spectral(G, n_clusters=10)

# Параметры обучения GNN
gnn_model, predictions = train_gnn_model(data, gnn_model, epochs=100, lr=0.01)
```

## Возможные проблемы

### Ошибка установки PyTorch Geometric
Если возникают проблемы с установкой PyTorch Geometric, попробуйте:
1. Установить PyTorch отдельно: `pip install torch`
2. Затем установить зависимости PyTorch Geometric по одной

### Ошибка с igraph/leidenalg
На Windows может потребоваться установка компилятора C++. Альтернатива - использовать предкомпилированные wheels:
```bash
py -m pip install python-igraph leidenalg --only-binary :all:
```

### Проблемы с памятью
Если граф слишком большой, уменьшите количество узлов:
```python
G, labels = generate_synthetic_twitter_graph(n_people=200, n_bots=50)
```

## Лицензия

Проект создан в образовательных целях.

## Автор

Senior Data Scientist - Эксперт по анализу графов и кибербезопасности
