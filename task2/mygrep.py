import sys
from gensim.models import Word2Vec
import os

def load_model(path="w2v_model.bin"):
    """Загрузка обученной модели Word2Vec."""
    if not os.path.exists(path):
        print(f"Файл модели '{path}' не найден! Поместите модель рядом с mygrep.py.")
        sys.exit(1)
    return Word2Vec.load(path)

def get_similar_words(model, word, topn=10):
    """Возвращает похожие слова по смыслу."""
    if word not in model.wv:
        print(f"Слово '{word}' отсутствует в словаре модели (OOV).")
        return []
    return [w for w, _ in model.wv.most_similar(word, topn=topn)]

def search_in_file(filepath, keywords):
    """Выводит строки файла, содержащие любое слово из keywords."""
    if not os.path.exists(filepath):
        print(f"Файл '{filepath}' не найден!")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            low = line.lower()
            if any(k in low for k in keywords):
                print(line.strip())

def main():
    if len(sys.argv) != 3:
        print("Использование: python mygrep.py data.txt \"слово\"")
        sys.exit(1)

    filepath = sys.argv[1]
    query = sys.argv[2].lower()

    print("Загрузка модели Word2Vec...")
    model = load_model()

    print(f"Поиск слов, похожих на '{query}'...")
    similar = get_similar_words(model, query)
    print("Похожие слова:", similar)

    # Все слова для поиска: исходное + похожие
    keywords = set([query] + similar)

    print("\nРезультаты поиска:")
    search_in_file(filepath, keywords)

if __name__ == "__main__":
    main()
