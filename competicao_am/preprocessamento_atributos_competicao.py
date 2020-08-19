import pandas as pd
from typing import List, Tuple

from base_am.preprocessamento_atributos import BagOfWords, BagOfItems


class DataframePreprocessing:
    def __init__(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        # Attributes
        self._df_treino = df_treino.copy()
        self._df_data_to_predict = df_data_to_predict.copy()
        self._col_classe = col_classe

        # Class to number
        self.dic_int_to_nom_classe = {}
        self.dic_nom_classe_to_int = {}

        # Processing
        self._process()

    @staticmethod
    def remove_id(*args: Tuple[pd.DataFrame]):
        dfs = (df.drop(columns='id') for df in args)
        return dfs[0] if len(dfs) == 1 else dfs

    def _process(self):
        """
        Aplica operações globais ao dataframe
        """
        pass
        # self._df_treino = self._df_treino.drop(columns=['id'])
        # self._df_data_to_predict = self._df_treino.drop(columns=['id'])

    def class_to_number(self, y):
        """
        Transforma classes em um mapa númerico
        Os mapeamentos são salvos em dic_int_to_nom_classe e dic_nom_classe_to_int
        """
        arr_int_y = []

        for rotulo_classe in y:
            if rotulo_classe not in self.dic_nom_classe_to_int:
                int_new_val_classe = len(self.dic_nom_classe_to_int.keys())
                self.dic_nom_classe_to_int[rotulo_classe] = int_new_val_classe
                self.dic_int_to_nom_classe[int_new_val_classe] = rotulo_classe

            arr_int_y.append(self.dic_nom_classe_to_int[rotulo_classe])

        return arr_int_y

    @property
    def df_treino(self) -> pd.DataFrame:
        return self._df_treino

    @property
    def df_data_to_predict(self) -> pd.DataFrame:
        return self._df_data_to_predict

    @property
    def col_classe(self) -> str:
        return self._col_clase

    @property
    def x(self):
        x_treino = self.df_treino.drop(self.col_classe, axis=1)
        x_to_predict = self.df_data_to_predict

        if self.col_classe in self.df_data_to_predict.columns:
            x_to_predict = self.df_data_to_predict.drop(
                self.col_classe, axis=1)

        return x_treino, x_to_predict

    @property
    def y(self):
        y_treino = self.class_to_number(self.df_treino[self.col_classe])
        y_to_predict = None

        if self.col_classe in self.df_data_to_predict.columns:
            y_to_predict = self.class_to_number(
                self.df_data_to_predict[self.col_classe])

        return y_treino, y_to_predict

    def nom_classe(self, classe: int) -> str:
        return self.dic_int_to_nom_classe[classe]

    def int_classe(self, classe: str) -> int:
        return self.dic_nom_classe_to_int[classe]

    def bag_of_items(self, items: List[str], min_occur: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cria uma bag of items a partir dos items passados
        """

        x_treino, x_to_predict = self.x

        boi = BagOfItems(min_occur=min_occur)

        training_boi_df = boi.cria_bag_of_items(x_treino, items)
        predict_boi_df = boi.aplica_bag_of_items(
            x_to_predict, items)

        return training_boi_df, predict_boi_df

    def bag_of_words(self, column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cria uma bag of words a partir da coluna passada
        """
        bow = BagOfWords()

        training_bow_df = bow.cria_bow(self.df_treino, column)
        predict_bow_df = bow.aplica_bow(self.df_data_to_predict, column)

        return training_boi_df, predict_boi_df
