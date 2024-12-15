from pydantic import BaseModel
from typing import List

class PostGet(BaseModel):
    '''
    Класс для валидации ответа рекомендаций постаов для посетителя

        Атрибуты:
            id - идентификатор поста
            text - текстовое содержание поста
            topic - тематика поста

    '''
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class Response(BaseModel):
    '''
    Класс для валидации ответа при использовании a/b теста
        Атрибуты:
            exp_group - группа в которую определяем посетителя при a/b-тесте
            recommendations - данные постов по схеме PostGet

    '''
    exp_group: str
    recommendations: List[PostGet]
    class Config:
        orm_mode = True