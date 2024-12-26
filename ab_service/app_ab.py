import os
from fastapi import FastAPI, HTTPException
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
from schema import Response
from dotenv import load_dotenv
import yaml
from loguru import logger
import hashlib

app = FastAPI()

### конфиги
def configuration_yaml():
    with open("../params.yaml", "r") as f:
        return yaml.safe_load(f)
config = configuration_yaml()

### подключение к бд
load_dotenv()
conn_uri = os.environ["POSTGRES_URL"]
logger.info('путь подключения загружен')

### функции для выгрузки модели, данных, формирования групп теста
def get_model_path(path: str, model_type: str) -> str:
    """
    Функция для определения пути откуда берется модель для использования в сервисе
    Используется для возможности проверки сервиса в чекере Karpov
    """
    if os.environ.get("IS_LMS") == "1":
        if model_type == 'control':
            MODEL_PATH = '/workdir/user_input/model_control'
        elif model_type == 'test':
            MODEL_PATH = '/workdir/user_input/model_test'
        else:
            raise ValueError('unknown model')
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models(path: str, type: str):
    """Функция для загрузки модели"""
    model_path = get_model_path(path, type)
    file_model = CatBoostClassifier()
    file_model.load_model(model_path, format="cbm")
    return file_model


def batch_load_sql(query: str) -> pd.DataFrame:
    """
    Функция для выгрузки больших данных из SQL без большой нагрузки памяти
    Выгрузка будет производиться кусками (чанками)
    Размер чанка в конфиге yaml, как и изначальный 200000
    """

    engine = create_engine(conn_uri)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=config['CHUNKSIZE']):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_like_posts() -> pd.DataFrame:
    """
    Функция для выгрузки постов, которые уже лайкали пользователи
    Чтобы их отсеять в сервисе для пользователя при составлении рекомендаций
    """
    like_posts = batch_load_sql("""SELECT distinct post_id, user_id 
                                        FROM feed_data 
                                        WHERE action = 'like'
                                        """)
    return like_posts

def load_post_features_control()-> pd.DataFrame:
    """
    Функция для выгрузки подготовленных фичей постов для контрольной модели
    """
    post_features = batch_load_sql("""SELECT * 
                                        FROM "al-smoljakov_features_lesson_22_port_1" """)
    return post_features

def load_post_features_test()-> pd.DataFrame:
    """
    Функция для выгрузки подготовленных фичей постов для тестовой модели
    """
    post_features = batch_load_sql("""SELECT * 
                                        FROM "al-smoljakov_features_lesson_22_port_2" """)
    return post_features

def load_features() -> pd.DataFrame:
    """
    Функция для выгрузки фичей пользователей
    """
    user_features = pd.read_sql("""
                                        SELECT
                                            user_id,
                                            gender,
                                            age,
                                            country,
                                            city,
                                            exp_group
                                        FROM user_data
                                    """,
                                con=conn_uri)
    return user_features



def get_exp_group(user_id: int) -> str:
    """
    Функция для формирования групп теста
    В конфиге соль и количество групп
    """
    exp_group = int(hashlib.md5((str(user_id) + config['SALTNAME']).encode()).hexdigest(), 16) % config['NUMBER_GROUP']
    if exp_group == 0:
        name_group = 'control'
    elif exp_group == 1:
        name_group = 'test'
    else:
        raise ValueError('check param number_group')
    return name_group


### выгружаем модели
model_control = load_models("./models/catboost_model_1.cbm", "control")
logger.info("контрольная модель загружена")
model_test = load_models("./models/catboost_model_2.cbm", "test")
logger.info("тестовая модель загружена")

### выгружаем залайканные посты у пользователей
df_like_posts = load_like_posts()
logger.info("посты-лайки на пользователя выгружены")

### выгружаем фичи постов
df_post_features_control = load_post_features_control()
df_post_features_test = load_post_features_test()
logger.info("фичи постов выгружены")

### выгружаем фичи пользователей
df_user_features = load_features()
logger.info("фичи пользователей выгружены")


### Функция по формированию рекомендаций вынесена отдельно от эндпоинта
def recommended_posts(id: int, time: datetime, user_group: str, limit: int = 5) -> Response:
    """
    Функция на вход получает запрашиваемые данные:
        id пользователя, время запросо постов, количество постов для рекомендации (выдаем по умолчанию 5)
    На основе выруженной модели и фичей формирует рекомендации для запрашиваемого пользователя в момент времени
    """

    ### из фичей постов формируем df c текстами постов (понадобятся в конце при формировании ответа)
    df_content = df_post_features_control[['post_id', 'text', 'topic']]

    ### из фичей пользователей забираем данные для текущего запрощенного id
    ### удаляем колонку user_id
    df_user = df_user_features.loc[df_user_features.user_id == id]
    df_user = df_user.drop('user_id', axis=1)

    ### отдельный df c фичами которые пойдут в модель в зависимости от группы теста
    if user_group == 'control':
        df_post = df_post_features_control.drop('text', axis=1)
    elif user_group == 'test':
        df_post = df_post_features_test.drop('text', axis=1)
    else:
        raise ValueError('unknown group')

    ### объединяем фичи постов и юзеров с помозью кросс джойна
    ### получается на одного юзера данные по всем постам
    df_input = pd.merge(df_post, df_user, how='cross').set_index('post_id')
    ### добавляем фичи по времени
    df_input['hour_cat'] = time.hour
    df_input['day_of_week_cat'] = time.weekday()
    df_input['month_cat'] = time.month

    ### применяем фичи в модели
    ### формируем колонку с вероятностью принадлежности объекта классу лайка
    if user_group == "control":
        df_input['predict'] = model_control.predict_proba(df_input)[:, 1]
        logger.info("model_control")
    else:
        df_input['predict'] = model_test.predict_proba(df_input)[:, 1]
        logger.info("model_test")

    ### выделяем посты, которые юзер уже лайкнул
    like_posts_user = df_like_posts[df_like_posts['user_id'] == id].post_id.values

    ### фильтруем посты которые юзер НЕ лайкал
    filter_out_df = df_input[~df_input.index.isin(like_posts_user)]
    ### сортируем посты по вероятности принадлежности лайку, формируем список из id постов
    rec_posts = filter_out_df.sort_values('predict', ascending=False)[:limit].index.to_list()

    ### из df с текстами забираем рекомендуемые посты, и приводим их к виду List[PostGet]
    recom_posts_response = df_content[df_content.post_id.isin(rec_posts)].rename(columns={"post_id": "id"}).to_dict(orient='records')

    result = {"exp_group": user_group,
                "recommendations": recom_posts_response}

    return result

### эндпоинт
@app.get("/post/recommendations/", response_model=Response)
def ab_recommended_post(id: int, time: datetime, limit: int = 5) -> Response:
    ### Проверяем существует ли id юзера на случай, если переданы неверные данные, при успехе передаем данные в функцию формирования рекомендаций
    if id in df_user_features['user_id'].values:
        ### формируем группу для теста и передаем в итоговую функцию
        user_group = get_exp_group(id)
        result = recommended_posts(id, time, user_group, limit)
        return result
    else:
        raise HTTPException(404, "user id not found")