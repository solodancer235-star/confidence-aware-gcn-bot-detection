"""
Прототип системы обнаружения бот-сетей в Twitter
Автор: Senior Data Scientist
Дата: 2024

Эта система использует комбинацию графовых методов, семантического анализа
и глубокого обучения для обнаружения бот-сетей в социальных сетях.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Библиотеки для машинного обучения
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import leidenalg
import igraph as ig

# PyTorch и PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

# ============================================================================
# ЭТАП 1: ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ
# ============================================================================

def generate_synthetic_twitter_graph(n_people=400, n_bots=100, seed=42):
    """
    Генерирует синтетический направленный граф Twitter с людьми и ботами.
    
    Параметры:
    -----------
    n_people : int
        Количество узлов-людей (по умолчанию 400)
    n_bots : int
        Количество узлов-ботов (по умолчанию 100)
    seed : int
        Seed для воспроизводимости результатов
    
    Возвращает:
    -----------
    G : networkx.DiGraph
        Направленный граф с узлами и их атрибутами
    labels : dict
        Словарь с метками узлов (0 = человек, 1 = бот)
    """
    np.random.seed(seed)
    G = nx.DiGraph()
    
    # Создаем узлы для людей
    people_nodes = list(range(n_people))
    for node in people_nodes:
        G.add_node(node, node_type='person')
    
    # Создаем узлы для ботов
    bot_nodes = list(range(n_people, n_people + n_bots))
    for node in bot_nodes:
        G.add_node(node, node_type='bot')
    
    # Генерируем органические связи для людей (Power-law распределение)
    # Используем Barabási-Albert модель для имитации органического роста
    # Создаем базовую структуру
    for i in range(5):
        for j in range(i+1, min(i+6, n_people)):
            if np.random.random() < 0.3:
                G.add_edge(i, j)
                if np.random.random() < 0.5:
                    G.add_edge(j, i)
    
    # Добавляем связи по принципу предпочтительного присоединения
    for new_node in range(5, n_people):
        # Выбираем несколько существующих узлов с вероятностью, пропорциональной их степени
        degrees = [G.degree(node) for node in range(new_node)]
        if sum(degrees) > 0:
            probs = np.array(degrees) / sum(degrees)
            targets = np.random.choice(new_node, size=min(3, new_node), 
                                     replace=False, p=probs)
            for target in targets:
                if np.random.random() < 0.4:
                    G.add_edge(new_node, target)
                if np.random.random() < 0.3:
                    G.add_edge(target, new_node)
    
    # Добавляем случайные связи между людьми
    for _ in range(n_people * 2):
        source = np.random.choice(people_nodes)
        target = np.random.choice(people_nodes)
        if source != target and np.random.random() < 0.1:
            G.add_edge(source, target)
    
    # Создаем 2 плотных кластера ботов (все связаны со всеми внутри кластера)
    n_bots_per_cluster = n_bots // 2
    cluster1_bots = bot_nodes[:n_bots_per_cluster]
    cluster2_bots = bot_nodes[n_bots_per_cluster:]
    
    # Внутренние связи в первом кластере ботов
    for i, bot1 in enumerate(cluster1_bots):
        for bot2 in cluster1_bots[i+1:]:
            G.add_edge(bot1, bot2)
            G.add_edge(bot2, bot1)
    
    # Внутренние связи во втором кластере ботов
    for i, bot1 in enumerate(cluster2_bots):
        for bot2 in cluster2_bots[i+1:]:
            G.add_edge(bot1, bot2)
            G.add_edge(bot2, bot1)
    
    # Боты следуют друг за другом между кластерами
    for bot1 in cluster1_bots[:5]:  # Некоторые боты из кластера 1
        for bot2 in cluster2_bots[:5]:  # следуют за ботами из кластера 2
            if np.random.random() < 0.7:
                G.add_edge(bot1, bot2)
                G.add_edge(bot2, bot1)
    
    # Боты следуют за некоторыми людьми (для маскировки)
    for bot in bot_nodes:
        # Каждый бот следует за 2-5 случайными людьми
        n_follows = np.random.randint(2, 6)
        targets = np.random.choice(people_nodes, size=n_follows, replace=False)
        for target in targets:
            G.add_edge(bot, target)
    
    # Генерируем текстовые атрибуты (твиты)
    # Тексты для людей - разнообразные
    people_topics = [
        "Сегодня отличная погода для прогулки в парке",
        "Только что посмотрел новый фильм, очень понравилось",
        "Работа над новым проектом идет полным ходом",
        "Встретил старого друга, было здорово поболтать",
        "Готовлю ужин, пробую новый рецепт",
        "Читаю интересную книгу о путешествиях",
        "Планирую отпуск на следующее лето",
        "Занимаюсь спортом, чувствую себя отлично",
        "Слушаю новую музыку, открываю для себя новые жанры",
        "Обсуждаю с коллегами интересные идеи",
        "Гуляю с собакой, она очень радуется",
        "Изучаю новый язык программирования",
        "Смотрю документальный фильм о природе",
        "Готовлюсь к важной презентации",
        "Встречаюсь с семьей на выходных"
    ]
    
    # Тексты для ботов - семантически похожие на тему крипто-скама
    bot_templates = [
        "Получите бесплатные биткоины прямо сейчас! Переходите по ссылке",
        "Уникальная возможность заработать на криптовалюте без вложений",
        "Быстрый заработок в интернете через криптовалюту гарантирован",
        "Получите двойной бонус при регистрации на криптобирже",
        "Секретный способ заработка на биткоинах раскрыт",
        "Заработайте тысячи долларов за один день на криптовалюте",
        "Бесплатные криптомонеты ждут вас, регистрируйтесь сейчас",
        "Проверенный метод заработка на криптовалюте без рисков",
        "Получите доступ к эксклюзивной крипто-стратегии",
        "Заработайте миллион на биткоинах за неделю"
    ]
    
    # Добавляем твиты людям
    for node in people_nodes:
        # Смешиваем несколько тем для разнообразия
        tweet_parts = np.random.choice(people_topics, size=2, replace=False)
        tweet = f"{tweet_parts[0]}. {tweet_parts[1]}."
        G.nodes[node]['tweets'] = tweet
    
    # Добавляем твиты ботам (семантически похожие)
    for node in bot_nodes:
        # Боты используют похожие шаблоны с небольшими вариациями
        base_template = np.random.choice(bot_templates)
        # Добавляем небольшие вариации
        variations = ["", " Не упустите шанс!", " Ограниченное предложение!", 
                     " Только сегодня!", " Проверено тысячами!"]
        variation = np.random.choice(variations)
        tweet = base_template + variation
        G.nodes[node]['tweets'] = tweet
    
    # Создаем словарь меток
    labels = {}
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'person':
            labels[node] = 0
        else:
            labels[node] = 1
    
    print(f"✓ Граф создан: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер")
    print(f"  - Людей: {n_people}, Ботов: {n_bots}")
    print(f"  - Средняя степень узла: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    return G, labels


# ============================================================================
# ЭТАП 2: СЕМАНТИЧЕСКИЙ СЛОЙ (SBERT)
# ============================================================================

def add_semantic_edges(G, model, similarity_threshold=0.85):
    """
    Добавляет семантические ребра между узлами на основе схожести их твитов.
    
    Параметры:
    -----------
    G : networkx.DiGraph
        Исходный граф
    model : SentenceTransformer
        Модель SBERT для вычисления эмбеддингов
    similarity_threshold : float
        Порог косинусного сходства для создания ребра (по умолчанию 0.85)
    
    Возвращает:
    -----------
    G : networkx.DiGraph
        Граф с добавленными семантическими ребрами
    n_semantic_edges : int
        Количество добавленных семантических ребер
    """
    print("\n" + "="*60)
    print("ЭТАП 2: Создание семантического слоя (SBERT)")
    print("="*60)
    
    # Извлекаем все твиты
    nodes = list(G.nodes())
    tweets = [G.nodes[node]['tweets'] for node in nodes]
    
    # Вычисляем эмбеддинги для всех твитов
    print("Вычисление эмбеддингов твитов...")
    embeddings = model.encode(tweets, show_progress_bar=True)
    
    # Вычисляем матрицу косинусного сходства
    print("Вычисление матрицы сходства...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Добавляем семантические ребра
    n_semantic_edges = 0
    semantic_edges = []
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i != j:  # Не создаем петли
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    # Проверяем, нет ли уже такого ребра
                    if not G.has_edge(node1, node2):
                        G.add_edge(node1, node2, edge_type='semantic', 
                                 similarity=similarity)
                        semantic_edges.append((node1, node2, similarity))
                        n_semantic_edges += 1
    
    print(f"✓ Добавлено {n_semantic_edges} семантических ребер")
    print(f"  Порог сходства: {similarity_threshold}")
    
    # Сохраняем эмбеддинги в атрибутах узлов для дальнейшего использования
    for i, node in enumerate(nodes):
        G.nodes[node]['embedding'] = embeddings[i]
    
    return G, n_semantic_edges


# ============================================================================
# ЭТАП 3: ОБНАРУЖЕНИЕ СООБЩЕСТВ
# ============================================================================

def detect_communities_leiden(G):
    """
    Обнаруживает сообщества в графе с помощью алгоритма Leiden.
    
    Параметры:
    -----------
    G : networkx.DiGraph
        Граф для анализа
    
    Возвращает:
    -----------
    communities : dict
        Словарь {node: community_id}
    n_communities : int
        Количество обнаруженных сообществ
    """
    print("\n" + "="*60)
    print("ЭТАП 3.1: Обнаружение сообществ (Leiden)")
    print("="*60)
    
    # Конвертируем NetworkX граф в igraph (Leiden работает с igraph)
    # Создаем неориентированную версию для Leiden
    G_undirected = G.to_undirected()
    
    # Создаем igraph из NetworkX
    edges = list(G_undirected.edges())
    if len(edges) == 0:
        print("Предупреждение: граф не содержит ребер")
        return {node: 0 for node in G.nodes()}, 1
    
    ig_graph = ig.Graph()
    ig_graph.add_vertices(G_undirected.number_of_nodes())
    ig_graph.add_edges(edges)
    
    # Применяем алгоритм Leiden
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    
    # Создаем словарь сообществ
    communities = {}
    for node_id in range(G_undirected.number_of_nodes()):
        communities[node_id] = partition.membership[node_id]
    
    n_communities = len(set(communities.values()))
    print(f"✓ Обнаружено {n_communities} сообществ")
    print(f"  Модулярность: {partition.modularity:.4f}")
    
    return communities, n_communities


def detect_anomalies_spectral(G, n_clusters=10):
    """
    Обнаруживает структурные аномалии с помощью Spectral Clustering.
    
    Параметры:
    -----------
    G : networkx.DiGraph
        Граф для анализа
    n_clusters : int
        Количество кластеров для Spectral Clustering
    
    Возвращает:
    -----------
    cluster_labels : dict
        Словарь {node: cluster_id}
    """
    print("\n" + "="*60)
    print("ЭТАП 3.2: Spectral Clustering для поиска аномалий")
    print("="*60)
    
    # Создаем матрицу смежности
    adj_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray()
    
    # Вычисляем нормализованную матрицу Лапласа
    # L = D^(-1/2) * (D - A) * D^(-1/2)
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    
    # Избегаем деления на ноль
    degree_matrix_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.sum(adj_matrix, axis=1), 1)))
    
    laplacian = degree_matrix - adj_matrix
    normalized_laplacian = degree_matrix_inv_sqrt @ laplacian @ degree_matrix_inv_sqrt
    
    # Применяем Spectral Clustering
    # Используем первые n_clusters собственных векторов
    eigenvals, eigenvecs = np.linalg.eigh(normalized_laplacian)
    
    # Берем собственные векторы, соответствующие наименьшим собственным значениям
    # (кроме первого, который всегда равен 0)
    k = min(n_clusters, len(eigenvals) - 1)
    if k > 0:
        features = eigenvecs[:, 1:k+1]
        
        # Нормализуем признаки
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Применяем K-means к собственным векторам
        spectral = SpectralClustering(n_clusters=n_clusters, 
                                    affinity='precomputed',
                                    random_state=42)
        
        # Создаем матрицу сходства из собственных векторов
        similarity = cosine_similarity(features_scaled)
        similarity = np.maximum(similarity, 0)  # Убираем отрицательные значения
        
        cluster_labels_array = spectral.fit_predict(similarity)
    else:
        cluster_labels_array = np.zeros(len(G.nodes()))
    
    # Создаем словарь меток кластеров
    cluster_labels = {}
    for i, node in enumerate(sorted(G.nodes())):
        cluster_labels[node] = int(cluster_labels_array[i])
    
    n_clusters_found = len(set(cluster_labels.values()))
    print(f"✓ Spectral Clustering завершен")
    print(f"  Найдено кластеров: {n_clusters_found}")
    
    return cluster_labels


# ============================================================================
# ЭТАП 4: GNN КЛАССИФИКАЦИЯ
# ============================================================================

class BotDetectionGCN(nn.Module):
    """
    Graph Convolutional Network для классификации ботов.
    
    Архитектура:
    - Входной слой: размерность признаков узлов
    - Скрытые слои: GCN слои с ReLU активацией
    - Выходной слой: бинарная классификация (бот/человек)
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=2):
        """
        Инициализация модели.
        
        Параметры:
        -----------
        input_dim : int
            Размерность входных признаков (эмбеддинги SBERT)
        hidden_dim : int
            Размерность скрытых слоев
        output_dim : int
            Размерность выхода (2 для бинарной классификации)
        num_layers : int
            Количество GCN слоев
        """
        super(BotDetectionGCN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Первый слой
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Скрытые слои
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Выходной слой
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        else:
            self.convs.append(GCNConv(input_dim, output_dim))
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index):
        """
        Прямой проход через сеть.
        
        Параметры:
        -----------
        x : torch.Tensor
            Матрица признаков узлов [num_nodes, input_dim]
        edge_index : torch.Tensor
            Индексы ребер [2, num_edges]
        
        Возвращает:
        -----------
        logits : torch.Tensor
            Логиты для каждого узла [num_nodes, output_dim]
        """
        # Применяем GCN слои
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Последний слой без активации (для логитов)
        x = self.convs[-1](x, edge_index)
        
        return x


def prepare_pytorch_geometric_data(G, labels):
    """
    Конвертирует NetworkX граф в формат PyTorch Geometric.
    
    Параметры:
    -----------
    G : networkx.DiGraph
        Граф NetworkX
    labels : dict
        Словарь меток узлов
    
    Возвращает:
    -----------
    data : torch_geometric.data.Data
        Данные в формате PyTorch Geometric
    """
    # Извлекаем эмбеддинги из атрибутов узлов
    node_features = []
    node_labels = []
    
    for node in sorted(G.nodes()):
        if 'embedding' in G.nodes[node]:
            node_features.append(G.nodes[node]['embedding'])
        else:
            # Если эмбеддинг отсутствует, создаем нулевой вектор
            # (не должно произойти, но на всякий случай)
            node_features.append(np.zeros(384))  # Размерность all-MiniLM-L6-v2
        
        node_labels.append(labels[node])
    
    # Конвертируем в тензоры
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long)
    
    # Создаем edge_index из графа
    edge_index = []
    for edge in G.edges():
        edge_index.append([edge[0], edge[1]])
    
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Создаем объект Data
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data


def train_gnn_model(data, model, epochs=100, lr=0.01):
    """
    Обучает GNN модель для классификации ботов.
    
    Параметры:
    -----------
    data : torch_geometric.data.Data
        Данные графа
    model : BotDetectionGCN
        Модель GNN
    epochs : int
        Количество эпох обучения
    lr : float
        Скорость обучения
    
    Возвращает:
    -----------
    model : BotDetectionGCN
        Обученная модель
    predictions : torch.Tensor
        Предсказания модели для всех узлов
    """
    print("\n" + "="*60)
    print("ЭТАП 4: Обучение GNN модели")
    print("="*60)
    
    # Разделяем данные на обучающую и тестовую выборки
    n_nodes = data.x.size(0)
    n_train = int(0.8 * n_nodes)
    
    indices = torch.randperm(n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[indices[:n_train]] = True
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask[indices[n_train:]] = True
    
    # Оптимизатор и функция потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Обучение
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            # Вычисляем точность на обучающей выборке
            pred = out[train_mask].argmax(dim=1)
            train_acc = (pred == data.y[train_mask]).sum().item() / train_mask.sum().item()
            
            # Вычисляем точность на тестовой выборке
            model.eval()
            with torch.no_grad():
                test_out = model(data.x, data.edge_index)
                test_pred = test_out[test_mask].argmax(dim=1)
                test_acc = (test_pred == data.y[test_mask]).sum().item() / test_mask.sum().item()
            model.train()
            
            print(f"Эпоха {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    # Финальные предсказания
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
        predictions = predictions.argmax(dim=1)
    
    # Вычисляем финальную точность
    final_acc = (predictions == data.y).sum().item() / len(data.y)
    print(f"\n✓ Обучение завершено")
    print(f"  Финальная точность: {final_acc:.4f}")
    
    return model, predictions


# ============================================================================
# ЭТАП 5: ВИЗУАЛИЗАЦИЯ
# ============================================================================

def visualize_graph(G, predictions, labels, communities=None, save_path='bot_detection_graph.png'):
    """
    Визуализирует граф с цветовой кодировкой предсказаний модели.
    
    Параметры:
    -----------
    G : networkx.DiGraph
        Граф для визуализации
    predictions : torch.Tensor или dict
        Предсказания модели (бот/человек)
    labels : dict
        Истинные метки узлов
    communities : dict, optional
        Словарь сообществ для дополнительной визуализации
    save_path : str
        Путь для сохранения изображения
    """
    print("\n" + "="*60)
    print("ЭТАП 5: Визуализация графа")
    print("="*60)
    
    # Конвертируем предсказания в словарь, если это тензор
    if isinstance(predictions, torch.Tensor):
        pred_dict = {}
        for i, node in enumerate(sorted(G.nodes())):
            pred_dict[node] = predictions[i].item()
    else:
        pred_dict = predictions
    
    # Создаем раскраску узлов
    node_colors = []
    node_sizes = []
    
    for node in sorted(G.nodes()):
        true_label = labels[node]
        pred_label = pred_dict[node]
        
        # Цвета:
        # Зеленый - правильно предсказанный человек
        # Красный - правильно предсказанный бот
        # Желтый - неправильно предсказанный человек (ложная тревога)
        # Оранжевый - неправильно предсказанный бот (пропуск)
        
        if true_label == 0 and pred_label == 0:  # Человек предсказан как человек
            node_colors.append('#2ecc71')  # Зеленый
        elif true_label == 1 and pred_label == 1:  # Бот предсказан как бот
            node_colors.append('#e74c3c')  # Красный
        elif true_label == 0 and pred_label == 1:  # Человек предсказан как бот
            node_colors.append('#f39c12')  # Оранжевый (ложная тревога)
        else:  # Бот предсказан как человек
            node_colors.append('#f1c40f')  # Желтый (пропуск)
        
        # Размер узла зависит от степени
        degree = G.degree(node)
        node_sizes.append(50 + degree * 2)
    
    # Создаем позиции узлов с помощью spring layout
    print("Вычисление позиций узлов...")
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Рисуем граф
    plt.figure(figsize=(16, 12))
    
    # Рисуем ребра (сделаем их полупрозрачными)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, arrows=False)
    
    # Рисуем узлы
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8)
    
    # Добавляем легенду
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Человек (правильно)'),
        mpatches.Patch(color='#e74c3c', label='Бот (правильно)'),
        mpatches.Patch(color='#f39c12', label='Человек → Бот (ложная тревога)'),
        mpatches.Patch(color='#f1c40f', label='Бот → Человек (пропуск)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.title('Обнаружение бот-сетей в Twitter\n'
              f'Всего узлов: {G.number_of_nodes()}, Ребер: {G.number_of_edges()}',
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Сохраняем изображение
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Граф сохранен в файл: {save_path}")
    
    # Вычисляем метрики
    correct = sum(1 for node in G.nodes() 
                 if labels[node] == pred_dict[node])
    accuracy = correct / len(G.nodes())
    
    # Подсчитываем ботов
    true_bots = sum(1 for label in labels.values() if label == 1)
    pred_bots = sum(1 for pred in pred_dict.values() if pred == 1)
    
    print(f"\nМетрики классификации:")
    print(f"  Точность: {accuracy:.4f} ({correct}/{len(G.nodes())})")
    print(f"  Истинных ботов: {true_bots}")
    print(f"  Предсказанных ботов: {pred_bots}")
    
    plt.show()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """
    Главная функция, которая запускает весь pipeline обнаружения ботов.
    """
    print("="*60)
    print("СИСТЕМА ОБНАРУЖЕНИЯ БОТ-СЕТЕЙ В TWITTER")
    print("="*60)
    print("\nИнициализация...")
    
    # ЭТАП 1: Генерация синтетических данных
    print("\n" + "="*60)
    print("ЭТАП 1: Генерация синтетических данных")
    print("="*60)
    G, labels = generate_synthetic_twitter_graph(n_people=400, n_bots=100, seed=42)
    
    # ЭТАП 2: Семантический слой
    print("\nЗагрузка модели Sentence-Transformer...")
    model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
    G, n_semantic = add_semantic_edges(G, model_sbert, similarity_threshold=0.85)
    
    # ЭТАП 3: Обнаружение сообществ
    communities, n_communities = detect_communities_leiden(G)
    spectral_clusters = detect_anomalies_spectral(G, n_clusters=10)
    
    # Добавляем информацию о сообществах в граф
    for node in G.nodes():
        G.nodes[node]['leiden_community'] = communities[node]
        G.nodes[node]['spectral_cluster'] = spectral_clusters[node]
    
    # ЭТАП 4: GNN классификация
    print("\nПодготовка данных для PyTorch Geometric...")
    data = prepare_pytorch_geometric_data(G, labels)
    
    # Определяем размерность входных признаков
    input_dim = data.x.size(1)
    print(f"Размерность признаков: {input_dim}")
    
    # Создаем и обучаем модель
    gnn_model = BotDetectionGCN(input_dim=input_dim, hidden_dim=64, 
                                output_dim=2, num_layers=3)
    gnn_model, predictions = train_gnn_model(data, gnn_model, epochs=100, lr=0.01)
    
    # ЭТАП 5: Визуализация
    visualize_graph(G, predictions, labels, communities=communities,
                   save_path='bot_detection_graph.png')
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("="*60)
    
    return G, labels, predictions, communities


if __name__ == "__main__":
    # Запускаем основной pipeline
    G, labels, predictions, communities = main()


# ============================================================================
# ИНСТРУКЦИЯ ПО ЗАПУСКУ
# ============================================================================
"""
КРАТКАЯ ИНСТРУКЦИЯ ПО ЗАПУСКУ:

1. УСТАНОВКА ЗАВИСИМОСТЕЙ:
   ------------------------
   На Windows:
   py -m pip install -r requirements.txt
   
   На Linux/Mac:
   pip install -r requirements.txt
   
   Дополнительно для PyTorch Geometric (после установки основных зависимостей):
   py -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
   py -m pip install torch-geometric

2. ЗАПУСК ПРОГРАММЫ:
   ------------------
   На Windows:
   py bot_detection_system.py
   
   На Linux/Mac:
   python bot_detection_system.py

3. РЕЗУЛЬТАТЫ:
   ------------
   - Программа выведет прогресс выполнения каждого этапа
   - Создаст файл 'bot_detection_graph.png' с визуализацией графа
   - Покажет метрики классификации (точность, количество обнаруженных ботов)

4. ПАРАМЕТРЫ НАСТРОЙКИ:
   ---------------------
   В функции main() можно изменить:
   - n_people, n_bots: количество людей и ботов
   - similarity_threshold: порог семантического сходства (0.85)
   - n_clusters: количество кластеров для Spectral Clustering (10)
   - epochs: количество эпох обучения GNN (100)
   - lr: скорость обучения (0.01)

5. ВОЗМОЖНЫЕ ПРОБЛЕМЫ:
   --------------------
   - Если PyTorch Geometric не устанавливается, установите PyTorch отдельно
   - На Windows может потребоваться установка компилятора C++ для igraph
   - При нехватке памяти уменьшите n_people и n_bots
"""
