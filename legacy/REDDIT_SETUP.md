# Инструкция по настройке Reddit API для сбора данных

## Шаг 1: Создание Reddit приложения

1. Перейдите на https://www.reddit.com/prefs/apps
2. Войдите в свой аккаунт Reddit (или создайте новый)
3. Прокрутите вниз и нажмите кнопку **"create app"** или **"create another app"**
4. Заполните форму:
   - **Name**: Любое название (например, "BotDetector")
   - **Type**: Выберите **"script"**
   - **Description**: Любое описание (можно оставить пустым)
   - **About URL**: Можно оставить пустым
   - **Redirect URI**: `http://localhost:8080` (обязательно!)
5. Нажмите **"create app"**

## Шаг 2: Получение credentials

После создания приложения вы увидите:
- **client_id**: Строка под названием приложения (выглядит как случайные символы)
- **client_secret**: Секретный ключ (показан только один раз, сохраните его!)

## Шаг 3: Настройка скрипта

### Вариант 1: Переменные окружения (рекомендуется)

**Windows (PowerShell):**
```powershell
$env:REDDIT_CLIENT_ID="ваш_client_id"
$env:REDDIT_CLIENT_SECRET="ваш_client_secret"
$env:REDDIT_USER_AGENT="BotDetector/1.0 by /u/ваш_username"
```

**Windows (CMD):**
```cmd
set REDDIT_CLIENT_ID=ваш_client_id
set REDDIT_CLIENT_SECRET=ваш_client_secret
set REDDIT_USER_AGENT=BotDetector/1.0 by /u/ваш_username
```

**Linux/Mac:**
```bash
export REDDIT_CLIENT_ID="ваш_client_id"
export REDDIT_CLIENT_SECRET="ваш_client_secret"
export REDDIT_USER_AGENT="BotDetector/1.0 by /u/ваш_username"
```

### Вариант 2: Прямое редактирование кода

Откройте `reddit_data_collector.py` и измените значения в `REDDIT_CONFIG`:

```python
REDDIT_CONFIG = {
    'client_id': 'ваш_client_id',
    'client_secret': 'ваш_client_secret',
    'user_agent': 'BotDetector/1.0 by /u/ваш_username'
}
```

## Шаг 4: Установка зависимостей

```bash
py -m pip install praw pandas
```

или

```bash
py -m pip install -r requirements.txt
```

## Шаг 5: Запуск скрипта

```bash
py reddit_data_collector.py
```

## Что делает скрипт

1. **Подключается к Reddit API** используя ваши credentials
2. **Собирает 100 последних постов** из r/CryptoCurrency
3. **Собирает комментарии** к каждому посту
4. **Создает графовую структуру:**
   - **nodes.csv**: Список всех пользователей с их активностью
   - **edges.csv**: Связи между пользователями (кто на чей комментарий ответил)

## Формат выходных файлов

### nodes.csv
Содержит информацию о пользователях:
- `id`: Имя пользователя
- `label`: Имя пользователя (для визуализации)
- `posts_count`: Количество созданных постов
- `comments_count`: Количество комментариев
- `total_score`: Суммарный score всех постов и комментариев
- `first_seen`: Дата первого появления
- `last_seen`: Дата последнего появления
- `total_activity`: Общая активность (посты + комментарии)

### edges.csv
Содержит связи между пользователями:
- `source`: Автор ответа
- `target`: Автор исходного комментария/поста
- `type`: Тип связи (reply_to_comment или reply_to_post)
- `weight`: Количество взаимодействий между пользователями
- `total_score`: Суммарный score комментариев
- `timestamp`: Время создания связи

## Ограничения Reddit API

- **Rate limit**: 60 запросов в минуту
- Скрипт автоматически добавляет задержки для соблюдения лимитов
- Для больших объемов данных может потребоваться время

## Устранение проблем

### Ошибка "Invalid credentials"
- Проверьте правильность client_id и client_secret
- Убедитесь, что выбрали тип приложения "script"

### Ошибка "403 Forbidden"
- Проверьте правильность user_agent
- Убедитесь, что приложение создано правильно

### Ошибка "Rate limit exceeded"
- Скрипт автоматически обрабатывает это, но если проблема сохраняется:
  - Увеличьте задержки в коде (time.sleep)
  - Уменьшите количество постов (NUM_POSTS)

## Безопасность

⚠️ **ВАЖНО**: Никогда не публикуйте ваши credentials в открытом доступе!
- Используйте переменные окружения вместо хардкода
- Не коммитьте файлы с credentials в Git
- Добавьте `.env` в `.gitignore` если используете файлы конфигурации
