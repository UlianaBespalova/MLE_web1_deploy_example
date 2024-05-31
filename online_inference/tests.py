import unittest

from fastapi.testclient import TestClient
from online_inference.app import app


class TestOnlineInference(unittest.TestCase):
    # Функция, которая выполняется перед каждым тестом
    # Подготавливаем данные, на которых будем тестировать приложение
    def setUp(self):
        self.data_to_predict = {
            "features_names": [
                'Date',
                'Location',
                'WindGustDir',
                'WindDir9am',
                'WindDir3pm',
                'MinTemp',
                'MaxTemp',
                'Rainfall',
                'Evaporation',
                'Sunshine',
                'WindGustSpeed',
                'WindSpeed9am',
                'WindSpeed3pm',
                'Humidity9am',
                'Humidity3pm',
                'Pressure9am',
                'Pressure3pm',
                'Cloud9am',
                'Cloud3pm',
                'Temp9am',
                'Temp3pm',
                'RainToday'
            ],
            "data": [
                ['2021-03-23', 'Cobar', 'W', 'W', 'W', 15, 18, 0.5, 3, 7, 40, 40, 40, 80, 80, 1000, 1000, 9, 9, 20, 20, 'Yes'],
                ['2021-03-23', 'Cobar', 'W', 'W', 'W', 15, 18, 0.5, 3, 7, 40, 40, 40, 80, 80, 1000, 1000, 9, 9, 20, 20, 'Yes'],
                ['2028-03-23', 'Albury', 'ENE', 'ENE', 'ENE', 15, 18, 0.9, 3, 7, 40, 40, 40, 90, 99, 1000, 1000, 9, 9, 20, 20, 'No'],
            ],
            "model": "lgbm",
        }

    # Проверяем работу корневой страницы. Ожидаем получить код 200 и сообщение "Predictor is alive"
    def test_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "Predictor is alive :)")

    def test_predict_empty_data(self):
        response = client.get("/predict")
        self.assertEqual(response.status_code, 405)

    # Проверка случаев, когда данные в post-запросе не проходят валидацию
    def test_predict_error_data(self):
        response = client.post("/predict", json={})
        self.assertEqual(response.status_code, 400) # Пустой json

        error_data = self.data_to_predict
        error_data["model"] = "randon model"
        response = client.post("/predict", json=error_data) # Неизвестный тип модели
        self.assertEqual(response.status_code, 400)

        error_data = self.data_to_predict
        error_data["data"] = [[20, 0.7, 3, 4, "36", 32, 99, 977, 3, 26],
                              [20, 0.7, 3, 4, 36, 32, 99, 977, 3, 26]]
        response = client.post("/predict", json=error_data) # Несоответствующие фичи
        self.assertEqual(response.status_code, 400)

        error_data = self.data_to_predict
        error_data["data"] = [[1]]
        response = client.post("/predict", json=error_data) # Неверный формат фич
        self.assertEqual(response.status_code, 400)

        error_data = self.data_to_predict
        error_data["features_names"] = ["hello", "world"]
        response = client.post("/predict", json=error_data) # Неверные название фич
        self.assertEqual(response.status_code, 400)

    # Проверяем ответ модели на правильные данные
    def test_predict_ok(self):
        with TestClient(app) as client:
            response = client.post("/predict", json=self.data_to_predict)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()['predicted_values']), 3) # Ожидаем 3 значения
            self.assertAlmostEqual(response.json()['predicted_values'][0], response.json()['predicted_values'][1])
            self.assertAlmostEqual(response.json()['predicted_values'][0], 0.036, delta=0.005)
            self.assertAlmostEqual(response.json()['predicted_values'][2], 0.71, delta=0.005)

    # Тест для /will_it_rain
    def test_will_it_rain_ok(self):
        with TestClient(app) as client:
            response = client.post("/will_it_rain", json=self.data_to_predict)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()['predicted_values']), 3)
            self.assertEqual(response.json()['predicted_values'][0], response.json()['predicted_values'][1])
            self.assertEqual(response.json()['predicted_values'][0], 0.0)
            self.assertEqual(response.json()['predicted_values'][2], 1.0)


if __name__ == '__main__':
    client = TestClient(app)
    unittest.main()
