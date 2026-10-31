from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity_matrix(sentences):
    """
    Вычисляет матрицу сходства для списка из 5 предложений используя модель Sentence-Transformer.
    
    Args:
        sentences (list): Список из 5 предложений (строк)
    
    Returns:
        numpy.ndarray: Матрица сходства размером 5x5, где каждый элемент [i][j] 
                      представляет косинусное сходство между предложениями i и j.
    
    Raises:
        ValueError: Если количество предложений не равно 5
    """
    if len(sentences) != 5:
        raise ValueError("Функция принимает ровно 5 предложений")
    
    # Загружаем модель Sentence-Transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Получаем эмбеддинги для всех предложений
    embeddings = model.encode(sentences)
    
    # Вычисляем матрицу косинусного сходства
    similarity_matrix = cosine_similarity(embeddings)
    
    return similarity_matrix


# Пример использования
if __name__ == "__main__":
    # Пример списка из 5 предложений
    example_sentences = [
        "Машинное обучение - это область искусственного интеллекта.",
        "Искусственный интеллект включает в себя машинное обучение.",
        "Сегодня хорошая погода для прогулки.",
        "Прогулка на свежем воздухе полезна для здоровья.",
        "Нейронные сети используются в глубоком обучении."
    ]
    
    # Вычисляем матрицу сходства
    matrix = calculate_similarity_matrix(example_sentences)
    
    # Выводим результат
    print("Матрица сходства предложений:")
    print(matrix)
    print("\nМатрица сходства (округленная до 3 знаков):")
    print(np.round(matrix, 3))
