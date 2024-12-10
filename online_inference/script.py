import requests
import pandas as pd


# Функция для отправки post-запроса.
# Выбирает n случайных строк из файла weather.csv.gz, упаковывает в json и отправляет
# На указанный адрес
def main(ip = "127.0.0.1", port = 8000):
    dataset = pd.read_csv("data/weather.csv.gz").dropna().sample(n=20, random_state=42)
    dataset.drop(["RainTomorrow"], axis=1, inplace=True)
    data_json = {
        "data": dataset.values.tolist(),
        "features_names": dataset.columns.tolist(),
        "model": "lgbm",
    }
    response = requests.post(
        f"http://{ip}:{port}/predict",
        json=data_json,
    )
    print(f"Status code:\n{response.status_code}")
    print(f"Result:\n{response.json()}")


if __name__ == "__main__":
    main()
