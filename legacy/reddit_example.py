"""
Пример использования данных, собранных из Reddit
для графового анализа с использованием NetworkX
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_reddit_graph(nodes_file='nodes.csv', edges_file='edges.csv'):
    """
    Загружает данные из CSV файлов и создает граф NetworkX.
    
    Параметры:
    -----------
    nodes_file : str
        Путь к файлу с узлами
    edges_file : str
        Путь к файлу с ребрами
    
    Возвращает:
    -----------
    G : networkx.DiGraph
        Направленный граф взаимодействий
    nodes_df : pandas.DataFrame
        DataFrame с информацией о пользователях
    edges_df : pandas.DataFrame
        DataFrame с информацией о связях
    """
    # Загружаем данные
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    
    # Создаем граф
    G = nx.DiGraph()
    
    # Добавляем узлы
    for _, row in nodes_df.iterrows():
        G.add_node(row['id'], 
                  posts_count=row['posts_count'],
                  comments_count=row['comments_count'],
                  total_score=row['total_score'],
                  total_activity=row['total_activity'])
    
    # Добавляем ребра
    for _, row in edges_df.iterrows():
        if 'weight' in row:
            G.add_edge(row['source'], row['target'], 
                      weight=row['weight'],
                      type=row.get('type', 'unknown'))
        else:
            G.add_edge(row['source'], row['target'],
                      type=row.get('type', 'unknown'))
    
    print(f"Граф создан: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер")
    
    return G, nodes_df, edges_df


def analyze_graph(G, nodes_df):
    """
    Выполняет базовый анализ графа.
    
    Параметры:
    -----------
    G : networkx.DiGraph
        Граф для анализа
    nodes_df : pandas.DataFrame
        DataFrame с информацией о пользователях
    """
    print("\n" + "="*60)
    print("АНАЛИЗ ГРАФА")
    print("="*60)
    
    # Базовая статистика
    print(f"\nБазовая статистика:")
    print(f"  Узлов: {G.number_of_nodes()}")
    print(f"  Ребер: {G.number_of_edges()}")
    print(f"  Плотность: {nx.density(G):.6f}")
    
    # Степени узлов
    degrees = dict(G.degree())
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    print(f"\nСтепени узлов:")
    print(f"  Средняя степень: {sum(degrees.values()) / len(degrees):.2f}")
    print(f"  Средняя входящая степень: {sum(in_degrees.values()) / len(in_degrees):.2f}")
    print(f"  Средняя исходящая степень: {sum(out_degrees.values()) / len(out_degrees):.2f}")
    
    # Топ пользователей по активности
    print(f"\nТоп-10 пользователей по активности:")
    top_users = nodes_df.nlargest(10, 'total_activity')[['id', 'total_activity', 'posts_count', 'comments_count']]
    for _, row in top_users.iterrows():
        print(f"  {row['id']}: {row['total_activity']} активностей "
              f"({row['posts_count']} постов, {row['comments_count']} комментариев)")
    
    # Топ пользователей по входящим связям (на них чаще отвечают)
    print(f"\nТоп-10 пользователей по входящим связям (популярность):")
    top_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    for user, degree in top_in:
        print(f"  {user}: {degree} входящих связей")
    
    # Топ пользователей по исходящим связям (активные комментаторы)
    print(f"\nТоп-10 пользователей по исходящим связям (активность):")
    top_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    for user, degree in top_out:
        print(f"  {user}: {degree} исходящих связей")


def visualize_graph(G, nodes_df, output_file='reddit_graph.png', max_nodes=100):
    """
    Визуализирует граф взаимодействий.
    
    Параметры:
    -----------
    G : networkx.DiGraph
        Граф для визуализации
    nodes_df : pandas.DataFrame
        DataFrame с информацией о пользователях
    output_file : str
        Имя файла для сохранения
    max_nodes : int
        Максимальное количество узлов для визуализации (для производительности)
    """
    print(f"\nВизуализация графа...")
    
    # Если граф слишком большой, берем подграф самых активных пользователей
    if G.number_of_nodes() > max_nodes:
        print(f"  Граф слишком большой ({G.number_of_nodes()} узлов)")
        print(f"  Создаю подграф из {max_nodes} самых активных пользователей...")
        
        top_users = nodes_df.nlargest(max_nodes, 'total_activity')['id'].tolist()
        G_sub = G.subgraph(top_users).copy()
    else:
        G_sub = G
    
    # Вычисляем позиции узлов
    pos = nx.spring_layout(G_sub, k=1, iterations=50, seed=42)
    
    # Размер узлов зависит от активности
    node_sizes = []
    node_colors = []
    
    for node in G_sub.nodes():
        if node in nodes_df['id'].values:
            activity = nodes_df[nodes_df['id'] == node]['total_activity'].values[0]
            node_sizes.append(50 + activity * 5)
            
            # Цвет зависит от соотношения постов и комментариев
            posts = nodes_df[nodes_df['id'] == node]['posts_count'].values[0]
            comments = nodes_df[nodes_df['id'] == node]['comments_count'].values[0]
            if posts > comments:
                node_colors.append('#e74c3c')  # Красный - больше постов
            else:
                node_colors.append('#3498db')  # Синий - больше комментариев
        else:
            node_sizes.append(50)
            node_colors.append('#95a5a6')  # Серый
    
    # Рисуем граф
    plt.figure(figsize=(16, 12))
    
    # Рисуем ребра
    nx.draw_networkx_edges(G_sub, pos, alpha=0.2, width=0.5, arrows=True, arrowsize=10)
    
    # Рисуем узлы
    nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, 
                          node_color=node_colors, alpha=0.8)
    
    # Подписи для топ узлов
    top_nodes = sorted(G_sub.nodes(), 
                      key=lambda x: nodes_df[nodes_df['id'] == x]['total_activity'].values[0] 
                      if x in nodes_df['id'].values else 0, 
                      reverse=True)[:10]
    labels = {node: node for node in top_nodes}
    nx.draw_networkx_labels(G_sub, pos, labels, font_size=8, font_weight='bold')
    
    plt.title(f'Граф взаимодействий в r/CryptoCurrency\n'
              f'{G_sub.number_of_nodes()} узлов, {G_sub.number_of_edges()} ребер',
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Граф сохранен в {output_file}")
    plt.show()


if __name__ == "__main__":
    # Загружаем данные
    try:
        G, nodes_df, edges_df = load_reddit_graph()
        
        # Анализируем граф
        analyze_graph(G, nodes_df)
        
        # Визуализируем
        visualize_graph(G, nodes_df)
        
    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден - {e}")
        print("Сначала запустите reddit_data_collector.py для сбора данных")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
