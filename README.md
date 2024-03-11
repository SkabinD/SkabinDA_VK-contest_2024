# Описание

Моя реализация профильного задания стажировки VK по напралению ML-инженер (2024).

Файлы:

    ./src/main.py - загрузка модели и предикт на основе тестовых данных, в результате работы в консоль выводится NDCG скор, предсказанные классы сохраняются в predicted_labels.csv в корне проекта;

    ./src/train_script.py -  обучение и сохранение модели;

    ./src/utils/py - дополнительные функции для обработки данных и вывода метрик;

    ./src/EDA.ipynb - поверхностный анализ датасетов;

    ./src/experiments.ipynb - эксперименты с моделями.


# Использование

Для запуска ввести в командную строку из корневой папки проекта:

docker build -t vk_contest:latest .

docker run -p 5000:5000 vk_contest

В качестве результата работы в командную строку будет выведен NDCG скор. Предсказанные классы будут сохранены в predicted_labels.csv

# Результаты

Текущий NDCG скор: 0.5771802140113097