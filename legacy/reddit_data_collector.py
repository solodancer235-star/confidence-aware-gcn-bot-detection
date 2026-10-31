"""
Скрипт для сбора данных из Reddit сабреддита r/CryptoCurrency
Использует библиотеку PRAW для получения постов, комментариев и их авторов.
Сохраняет данные в формате nodes.csv и edges.csv для графового анализа.
"""

import praw
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
import time

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Reddit API credentials
# Получите их на https://www.reddit.com/prefs/apps
# Создайте новое приложение типа "script"
REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID', 'YOUR_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET', 'YOUR_CLIENT_SECRET'),
    'user_agent': os.getenv('REDDIT_USER_AGENT', 'BotDetector/1.0 by /u/YOUR_USERNAME')
}

# Параметры сбора данных
SUBREDDIT_NAME = 'CryptoCurrency'
NUM_POSTS = 100  # Количество последних постов для анализа
MAX_COMMENTS_PER_POST = 100  # Максимальное количество комментариев на пост

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ PRAW
# ============================================================================

def init_reddit_client():
    """
    Инициализирует клиент Reddit API.
    
    Возвращает:
    -----------
    reddit : praw.Reddit
        Объект клиента Reddit
    """
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CONFIG['client_id'],
            client_secret=REDDIT_CONFIG['client_secret'],
            user_agent=REDDIT_CONFIG['user_agent']
        )
        
        # Проверка подключения
        print(f"✓ Подключение к Reddit установлено")
        print(f"  Режим: {'Read-only' if reddit.read_only else 'Read-write'}")
        
        return reddit
    except Exception as e:
        print(f"✗ Ошибка при подключении к Reddit: {e}")
        print("\nУбедитесь, что вы установили правильные credentials:")
        print("1. Перейдите на https://www.reddit.com/prefs/apps")
        print("2. Создайте новое приложение типа 'script'")
        print("3. Установите переменные окружения или измените REDDIT_CONFIG")
        raise


# ============================================================================
# СБОР ДАННЫХ
# ============================================================================

def collect_posts_and_comments(reddit, subreddit_name, num_posts=100):
    """
    Собирает посты и комментарии из указанного сабреддита.
    
    Параметры:
    -----------
    reddit : praw.Reddit
        Клиент Reddit API
    subreddit_name : str
        Название сабреддита (без r/)
    num_posts : int
        Количество постов для анализа
    
    Возвращает:
    -----------
    posts_data : list
        Список словарей с данными постов
    comments_data : list
        Список словарей с данными комментариев
    """
    print(f"\n{'='*60}")
    print(f"СБОР ДАННЫХ ИЗ r/{subreddit_name}")
    print(f"{'='*60}")
    
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    comments_data = []
    
    print(f"\nПолучение {num_posts} последних постов...")
    
    post_count = 0
    for submission in subreddit.new(limit=num_posts):
        post_count += 1
        
        # Данные поста
        post_data = {
            'id': submission.id,
            'type': 'post',
            'author': str(submission.author) if submission.author else '[deleted]',
            'title': submission.title,
            'text': submission.selftext if hasattr(submission, 'selftext') else '',
            'score': submission.score,
            'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
            'num_comments': submission.num_comments,
            'url': submission.url
        }
        posts_data.append(post_data)
        
        if post_count % 10 == 0:
            print(f"  Обработано постов: {post_count}/{num_posts}")
        
        # Собираем комментарии
        print(f"  Сбор комментариев для поста {post_count}...")
        submission.comments.replace_more(limit=0)  # Убираем "load more comments"
        
        comment_count = 0
        for comment in submission.comments.list():
            if comment_count >= MAX_COMMENTS_PER_POST:
                break
            
            # Пропускаем удаленные комментарии
            if comment.author is None:
                continue
            
            comment_data = {
                'id': comment.id,
                'type': 'comment',
                'author': str(comment.author),
                'text': comment.body,
                'score': comment.score,
                'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                'post_id': submission.id,
                'post_author': post_data['author'],
                'parent_id': comment.parent_id,  # ID родительского комментария или поста
                'is_submission': comment.is_submission  # True если ответ на пост, False если на комментарий
            }
            comments_data.append(comment_data)
            comment_count += 1
        
        # Небольшая задержка, чтобы не превысить rate limit
        time.sleep(0.1)
    
    print(f"\n✓ Сбор данных завершен")
    print(f"  Постов собрано: {len(posts_data)}")
    print(f"  Комментариев собрано: {len(comments_data)}")
    
    return posts_data, comments_data


# ============================================================================
# ОБРАБОТКА ДАННЫХ ДЛЯ ГРАФА
# ============================================================================

def process_data_for_graph(posts_data, comments_data):
    """
    Обрабатывает собранные данные для создания графа.
    Создает узлы (пользователи) и ребра (ответы на комментарии).
    
    Параметры:
    -----------
    posts_data : list
        Данные постов
    comments_data : list
        Данные комментариев
    
    Возвращает:
    -----------
    nodes_df : pandas.DataFrame
        DataFrame с узлами (пользователями)
    edges_df : pandas.DataFrame
        DataFrame с ребрами (связями между пользователями)
    """
    print(f"\n{'='*60}")
    print("ОБРАБОТКА ДАННЫХ ДЛЯ ГРАФА")
    print(f"{'='*60}")
    
    # Собираем информацию о пользователях
    users_info = defaultdict(lambda: {
        'posts_count': 0,
        'comments_count': 0,
        'total_score': 0,
        'first_seen': None,
        'last_seen': None
    })
    
    # Обрабатываем посты
    for post in posts_data:
        author = post['author']
        if author != '[deleted]':
            users_info[author]['posts_count'] += 1
            users_info[author]['total_score'] += post['score']
            if users_info[author]['first_seen'] is None:
                users_info[author]['first_seen'] = post['created_utc']
            users_info[author]['last_seen'] = post['created_utc']
    
    # Обрабатываем комментарии
    for comment in comments_data:
        author = comment['author']
        if author != '[deleted]':
            users_info[author]['comments_count'] += 1
            users_info[author]['total_score'] += comment['score']
            if users_info[author]['first_seen'] is None:
                users_info[author]['first_seen'] = comment['created_utc']
            users_info[author]['last_seen'] = comment['created_utc']
    
    # Создаем DataFrame узлов
    nodes_list = []
    for user, info in users_info.items():
        nodes_list.append({
            'id': user,
            'label': user,
            'posts_count': info['posts_count'],
            'comments_count': info['comments_count'],
            'total_score': info['total_score'],
            'first_seen': info['first_seen'],
            'last_seen': info['last_seen'],
            'total_activity': info['posts_count'] + info['comments_count']
        })
    
    nodes_df = pd.DataFrame(nodes_list)
    print(f"✓ Создано узлов (пользователей): {len(nodes_df)}")
    
    # Создаем ребра (кто на чей комментарий ответил)
    edges_list = []
    
    # Создаем словарь для быстрого поиска комментариев по ID
    comments_by_id = {}
    for comment in comments_data:
        comments_by_id[comment['id']] = comment
    
    # Обрабатываем связи
    for comment in comments_data:
        comment_author = comment['author']
        parent_id = comment['parent_id']
        
        # Пропускаем удаленных авторов
        if comment_author == '[deleted]':
            continue
        
        # Определяем, на что отвечает комментарий
        if comment['is_submission']:
            # Ответ на пост
            parent_author = comment['post_author']
            edge_type = 'reply_to_post'
        else:
            # Ответ на комментарий
            # parent_id имеет формат "t1_xxxxx" для комментариев или "t3_xxxxx" для постов
            if parent_id.startswith('t1_'):
                # Это комментарий
                parent_comment_id = parent_id[3:]  # Убираем префикс "t1_"
                if parent_comment_id in comments_by_id:
                    parent_author = comments_by_id[parent_comment_id]['author']
                    edge_type = 'reply_to_comment'
                else:
                    continue  # Родительский комментарий не найден
            elif parent_id.startswith('t3_'):
                # Это пост
                parent_author = comment['post_author']
                edge_type = 'reply_to_post'
            else:
                continue
        
        # Пропускаем, если родительский автор удален
        if parent_author == '[deleted]':
            continue
        
        # Пропускаем самоссылки
        if comment_author == parent_author:
            continue
        
        # Добавляем ребро
        edges_list.append({
            'source': comment_author,
            'target': parent_author,
            'type': edge_type,
            'comment_id': comment['id'],
            'comment_text': comment['text'][:200],  # Первые 200 символов
            'comment_score': comment['score'],
            'timestamp': comment['created_utc']
        })
    
    edges_df = pd.DataFrame(edges_list)
    
    # Подсчитываем количество связей между каждой парой пользователей
    if len(edges_df) > 0:
        edges_summary = edges_df.groupby(['source', 'target']).agg({
            'comment_id': 'count',
            'comment_score': 'sum'
        }).reset_index()
        edges_summary.columns = ['source', 'target', 'weight', 'total_score']
        edges_df = edges_summary.merge(
            edges_df[['source', 'target', 'type', 'timestamp']].drop_duplicates(),
            on=['source', 'target'],
            how='left'
        )
    
    print(f"✓ Создано ребер (связей): {len(edges_df)}")
    
    return nodes_df, edges_df


# ============================================================================
# СОХРАНЕНИЕ ДАННЫХ
# ============================================================================

def save_to_csv(nodes_df, edges_df, nodes_file='nodes.csv', edges_file='edges.csv'):
    """
    Сохраняет узлы и ребра в CSV файлы.
    
    Параметры:
    -----------
    nodes_df : pandas.DataFrame
        DataFrame с узлами
    edges_df : pandas.DataFrame
        DataFrame с ребрами
    nodes_file : str
        Имя файла для узлов
    edges_file : str
        Имя файла для ребер
    """
    print(f"\n{'='*60}")
    print("СОХРАНЕНИЕ ДАННЫХ")
    print(f"{'='*60}")
    
    # Сохраняем узлы
    nodes_df.to_csv(nodes_file, index=False, encoding='utf-8-sig')
    print(f"✓ Узлы сохранены в {nodes_file}")
    print(f"  Количество узлов: {len(nodes_df)}")
    print(f"  Колонки: {', '.join(nodes_df.columns)}")
    
    # Сохраняем ребра
    if len(edges_df) > 0:
        edges_df.to_csv(edges_file, index=False, encoding='utf-8-sig')
        print(f"✓ Ребра сохранены в {edges_file}")
        print(f"  Количество ребер: {len(edges_df)}")
        print(f"  Колонки: {', '.join(edges_df.columns)}")
    else:
        print(f"⚠ Ребра не найдены, файл {edges_file} не создан")
    
    # Выводим статистику
    print(f"\nСтатистика:")
    if len(nodes_df) > 0:
        print(f"  Среднее количество постов на пользователя: {nodes_df['posts_count'].mean():.2f}")
        print(f"  Среднее количество комментариев на пользователя: {nodes_df['comments_count'].mean():.2f}")
        print(f"  Средний общий score: {nodes_df['total_score'].mean():.2f}")
    
    if len(edges_df) > 0:
        print(f"  Средний вес ребра: {edges_df['weight'].mean():.2f if 'weight' in edges_df.columns else 'N/A'}")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """
    Главная функция для сбора данных из Reddit.
    """
    print("="*60)
    print("СБОР ДАННЫХ ИЗ REDDIT")
    print("="*60)
    
    # Проверка credentials
    if (REDDIT_CONFIG['client_id'] == 'YOUR_CLIENT_ID' or 
        REDDIT_CONFIG['client_secret'] == 'YOUR_CLIENT_SECRET'):
        print("\n⚠ ВНИМАНИЕ: Необходимо настроить Reddit API credentials!")
        print("\nИнструкция:")
        print("1. Перейдите на https://www.reddit.com/prefs/apps")
        print("2. Нажмите 'create app' или 'create another app'")
        print("3. Выберите тип 'script'")
        print("4. Заполните название и описание (любые)")
        print("5. В поле 'redirect uri' укажите: http://localhost:8080")
        print("6. После создания вы получите client_id и client_secret")
        print("\nЗатем установите переменные окружения:")
        print("  set REDDIT_CLIENT_ID=your_client_id")
        print("  set REDDIT_CLIENT_SECRET=your_client_secret")
        print("  set REDDIT_USER_AGENT=YourAppName/1.0 by /u/your_username")
        print("\nИли измените значения в REDDIT_CONFIG в начале файла.")
        return
    
    try:
        # Инициализация клиента
        reddit = init_reddit_client()
        
        # Сбор данных
        posts_data, comments_data = collect_posts_and_comments(
            reddit, 
            SUBREDDIT_NAME, 
            NUM_POSTS
        )
        
        # Обработка данных для графа
        nodes_df, edges_df = process_data_for_graph(posts_data, comments_data)
        
        # Сохранение в CSV
        save_to_csv(nodes_df, edges_df)
        
        print("\n" + "="*60)
        print("СБОР ДАННЫХ ЗАВЕРШЕН УСПЕШНО")
        print("="*60)
        print(f"\nФайлы созданы:")
        print(f"  - nodes.csv: {len(nodes_df)} пользователей")
        print(f"  - edges.csv: {len(edges_df)} связей")
        print(f"\nДанные готовы для графового анализа!")
        
    except Exception as e:
        print(f"\n✗ Ошибка при выполнении: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
