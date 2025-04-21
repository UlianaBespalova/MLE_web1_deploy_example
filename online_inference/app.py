import logging
import sys
import uvicorn

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Импортируем вспомогательные функции из data_utils
from .data_utils import (
    InputData,
    OutputData,
    get_data,
    get_model,
)

# Настраиваем логгирование
logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
)
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

app = FastAPI()

model_lgbm = None # Переменная, в которой будет храниться модель

# Функция, срабатывающая при запуске сервера
@app.on_event("startup")
def startup():
    global model_lgbm
    model_path = "models/model_lgbm_classifier.pickle" # Загружаем модель
    try:
        model_lgbm = get_model(model_path) # Если успешно, пишем, что всё ок
        logger.info(msg="Model is loaded")
    except Exception as e:
        print(e)
        logger.error(f"model not found") # Иначе пишем предупреждение
        raise RuntimeError(f"model not found")

# При открытии корневой страницы пишем, что всё хорошо
@app.get("/")
def main():
    return "Predictor is alive :)"

# Функция, которая получает данные в post-запросе и возвращает скор
@app.post("/predict", response_model=OutputData)
def predict(request: InputData):
    data = get_data(request) # Парсим данные из запроса в DataFrame
    logger.info(msg=f"Data loaded")
    try:
        y_pred = model_lgbm.predict_proba(data)[:, 1] # Получаем предикт
    except Exception as e:
        raise HTTPException( # Если что-то идёт не так, выдаём ошибку и код 500
            status_code=500,
            detail="Error: something went wrong while prediction")

    logger.info(msg=f"Prediction finished. It's OK :) {y_pred}")
    return OutputData(predicted_values=y_pred) # Возвращаем результат


@app.get("/is_ready")
def is_ready():
    if model_lgbm:
        status = 200
        msg = "Model is ready ^_^"
    else:
        status = 500
        msg = "Model not found :("
    return JSONResponse(
        status_code = status,
        content = jsonable_encoder({"Message": msg})
    )




# Функция, срабатывающая при ошибке
# (если данные в post запросе имеют неправильный формат)
@app.exception_handler(RequestValidationError)
async def validate_data(
    _: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, # Сообщаем об ошибке и пишем, что именно пошло не так
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
