import pandas as pd
from typing import List, Tuple

from base_am.preprocessamento_atributos import BagOfWords, BagOfItems


def gerar_atributos_ator(df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame) -> pd.DataFrame:
    obj_bag_of_actors = BagOfItems(min_occur=3)
    df_treino_boa = obj_bag_of_actors.cria_bag_of_items(
        df_treino, ["ator_1", "ator_2", "ator_3", "ator_4", "ator_5"])
    df_data_to_predict_boa = obj_bag_of_actors.aplica_bag_of_items(
        df_data_to_predict, ["ator_1", "ator_2", "ator_3", "ator_4", "ator_5"])

    return df_treino_boa, df_data_to_predict_boa


def gerar_atributos_resumo(df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame) -> pd.DataFrame:
    bow_amostra = BagOfWords()
    df_bow_treino = bow_amostra.cria_bow(df_treino, "resumo")
    df_bow_data_to_predict = bow_amostra.aplica_bow(
        df_data_to_predict, "resumo")

    return df_bow_treino, df_bow_data_to_predict


class DataframePreprocessing:
    def __init__(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame):
        self.df_treino = df_treino
        self.df_data_to_predict = df_data_to_predict

    def _bag_of_items(self, items: List[str], min_occur: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        boi = BagOfItems(min_occur=min_occur)

        training_boi_df = boi.cria_bag_of_items(self.df_treino, items)
        predict_boi_df = boi.aplica_bag_of_items(
            self.df_data_to_predict, items)

        return training_boi_df, predict_boi_df

    def _bag_of_words(self, column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        bow = BagOfWords()

        training_bow_df = bow.cria_bow(self.df_treino, column)
        predict_bow_df = bow.aplica_bow(self.df_data_to_predict, column)

        return training_boi_df, predict_boi_df
