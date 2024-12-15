import os
from fastapi import FastAPI, HTTPException
from datetime import datetime
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
from schema import PostGet
from typing import List, Optional
from dotenv import load_dotenv
from config import config
from loguru import logger

app = FastAPI()

### подключение к бд
load_dotenv()
conn_uri = os.environ["POSTGRES_URL"]
logger.info('путь подключения загружен')

### функции для выгрузки модели и данных
def get_model_path(path: str) -> str:
    """
    Функция для определения пути откуда берется модель для использования в сервисе
    Используется для возможности проверки сервиса в чекере Karpov
    """

    if os.environ.get("IS_LMS") == "1":  
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    """Функция для загрузки модели"""
    model_path = get_model_path("catboost_model.cbm")
    file_model = CatBoostClassifier()
    file_model.load_model(model_path, format="cbm")
    return file_model


def batch_load_sql(query: str) -> pd.DataFrame:
    """
    Функкция для выгрузки больших данных из SQL без большой нагрузки памяти
    Выгрузка будет производиться кусками (чанками)
    Размер чанка в конфиге yaml, изначальный 200000
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

def load_post_features()-> pd.DataFrame:
    """
    Функция для выгрузки подготовленных фичей постов
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

### выгружаем модель
model = load_models()
logger.info("модель загружена")

### выгружаем залайканные посты у пользователей
df_like_posts = load_like_posts()
logger.info("посты-лайки на пользователя выгружены")

### выгружаем фичи постов
df_post_features = load_post_features()
logger.info("фичи постов выгружены")

### выгружаем фичи пользователей
df_user_features = load_features()
logger.info("фичи пользователей выгружены")

### функционал рекомендаций
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    """
    Функция на вход получает запрашиваемые данные:
        id пользователя, время запросо постов, количество постов для рекомендации (выдаем по умолчанию 5)
    На основе выруженной модели и фичей формирует рекомендации для запрашиваемого пользователя в момент времени
    """
    if id in df_user_features['user_id'].values:
        ### из фичей постов формируем df c текстами постов (понадобятся в конце при формировании ответа)
        df_content = df_post_features[['post_id', 'text', 'topic']]
        ### отдельный df c фичами которые пойдут в модель (удаляем ненужную колонку text)
        df_post = df_post_features.drop('text', axis=1)

        ### из фичей пользователей забираем данные для текущего запрощенного id
        ### удаляем колонку user_id
        df_user = df_user_features.loc[df_user_features.user_id == id]
        df_user = df_user.drop('user_id', axis=1)

        ### объединяем фичи постов и юзеров с помозью кросс джойна
        ### получается на одного юзера данные по всем постам
        df_input = pd.merge(df_post, df_user, how='cross').set_index('post_id')
        ### добавляем фичи по времени
        df_input['hour_cat'] = time.hour
        df_input['day_of_week_cat'] = time.weekday()
        df_input['month_cat'] = time.month

        ### применяем фичи в модели
        ### формируем колонку с вероятностью принадлежности объекта классу лайка
        df_input['predict'] = model.predict_proba(df_input)[:, 1]

        ### выделяем посты, которые юзер уже лайкнул
        like_posts_user = df_like_posts[df_like_posts['user_id'] == id].post_id.values

        ### фильтруем посты которые юзер НЕ лайкал
        filter_out_df = df_input[~df_input.index.isin(like_posts_user)]
        ### сортируем посты по вероятности принадлежности лайку, формируем список из id постов
        rec_posts = filter_out_df.sort_values('predict', ascending=False)[:limit].index.to_list()

        ### из df с текстами забираем рекомендуемые посты, и приводим их к виду List[PostGet]
        result = df_content[df_content.post_id.isin(rec_posts)].rename(columns={"post_id": "id"}).to_dict(orient='records')

        return result
    else:
        raise HTTPException(404, "user id not found")