import pandas as pd
from sklearn import preprocessing, compose
from typing import List, Tuple

from base_am.preprocessamento_atributos import BagOfWords, BagOfItems

import re
import string
import numpy as np
import json

sw_en = json.load(open('datasets/en_sw.json', 'r'))


class DataframePreprocessing:
    def __init__(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        # Attributes
        self.df_treino = df_treino.copy()
        self.df_data_to_predict = df_data_to_predict.copy()
        self._col_classe = col_classe

        # Class to number
        self.dic_int_to_nom_classe = {}
        self.dic_nom_classe_to_int = {}

    @staticmethod
    def remove_id(*args: Tuple[pd.DataFrame]):
        dfs = tuple(df.drop(columns='id')
                    if 'id' in df.columns else df for df in args)
        return dfs[0] if len(dfs) == 1 else dfs

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
    def col_classe(self) -> str:
        return self._col_classe

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

        # Concatenating the bag with the initial dataframe might give better results

        x_treino, x_to_predict = self.x

        bow = BagOfWords()

        training_bow_df = bow.cria_bow(x_treino, column)
        predict_bow_df = bow.aplica_bow(x_to_predict, column)

        return training_bow_df, predict_bow_df

    def cleaner(overview: str) -> str:
        if overview is None or overview is np.nan:
            return None

        # Separators
        separators_exp = r'[\-_]'
        # Other punctuation
        punctuation_tt = str.maketrans('', '', string.punctuation)
        # Garbage
        garbage_exp = r'[^A-z\s]'

        new_overview = overview
        new_overview = overview.lower()  # Normalize case
        # Normalize separators
        new_overview = re.sub(separators_exp, ' ', new_overview)
        new_overview = new_overview.translate(
            punctuation_tt)  # Remove punctuation
        new_overview = re.sub(garbage_exp, '', new_overview)

        new_overview = ' '.join(
            [w for w in new_overview.split() if len(w) > 0 and w not in sw_en])

        return new_overview if len(new_overview) != 0 else None

    def generate_dataframes(self) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
        """
        Cria os pares de dataframes a serem usados
        """

        """
        Possíveis DF's
        - limpo, sem nenhuma coluna que será transformada em BoW ou BoI
        - BoI, combinação das BoI
        - BoW, combinação das BoW
        - BoI Parametrizado, BoI e atributos padrões
        - BoW Parametrizado, BoW e atributos padrões
        - completo, todas as colunas (pode ser muito lento)
        """

        x_treino, x_to_predict = self.x

        """
        Colunas não usadas:
        - id
        - titulo
        - idioma_original
        - historia_original
        - adulto
        - data_de_estreia
        """

        colunas_a_remover = ['id', 'titulo',
                             'idioma_original', 'adulto', 'data_de_estreia']

        # Escalamento das colunas

        # orcamento, popularidade, receita e duracao
        to_scale = ['orcamento', 'popularidade', 'receita', 'duracao']
        scaler = preprocessing.StandardScaler()

        scaler.fit(x_treino[to_scale])
        x_treino[to_scale] = scaler.transform(x_treino[to_scale])
        x_to_predict[to_scale] = scaler.transform(x_to_predict[to_scale])

        # Criação da(s) BoI(s)

        items = ['ator_1', 'ator_2', 'ator_3', 'ator_4', 'ator_5',
                 'dirigido_por', 'escrito_por_1', 'escrito_por_2']

        boi_treino, boi_to_predict = self.bag_of_items(items)
        boi_treino, boi_to_predict = self.remove_id(boi_treino, boi_to_predict)
        boi_treino.columns = [
            f'boi_{c}' for c in boi_treino.columns]
        boi_to_predict.columns = [
            f'boi_{c}' for c in boi_to_predict.columns]

        # Limpeza do resumo

        x_treino.resumo = x_treino.resumo.apply(self.cleaner)
        x_to_predict.resumo = x_to_predict.resumo.apply(self.cleaner)

        # Criação da(s) BoW(s)

        bow = ['resumo']  # A relevância do título é questionável

        bow_resumo_treino, bow_resumo_to_predict = self.bag_of_words('resumo')
        bow_resumo_treino.columns = [
            f'resumo_{c}' for c in bow_resumo_treino.columns]
        bow_resumo_to_predict.columns = [
            f'resumo_{c}' for c in bow_resumo_to_predict.columns]

        # Criação dos dataframes

        def clean(df):
            return df.drop(columns=colunas_a_remover).select_dtypes(exclude=['object']).fillna(0)

        df_limpo_treino = clean(x_treino)
        df_limpo_to_predict = clean(x_to_predict)

        def c_concat(dfs):
            return pd.concat([df.reset_index() for df in dfs], axis=1)

        df_boi_treino = c_concat([df_limpo_treino, boi_treino])
        df_boi_to_predict = c_concat([df_limpo_to_predict, boi_to_predict])

        df_bow_treino = c_concat([df_limpo_treino, bow_resumo_treino])
        df_bow_to_predict = c_concat(
            [df_limpo_to_predict, bow_resumo_to_predict])

        df_completo_treino = c_concat(
            [df_limpo_treino, boi_treino, bow_resumo_treino])
        df_completo_to_predict = c_concat(
            [df_limpo_to_predict, boi_to_predict, bow_resumo_to_predict])

        dataframes = {
            'limpo': (df_limpo_treino, df_limpo_to_predict),
            'boi': (df_boi_treino, df_boi_to_predict),
            'bow': (df_bow_treino, df_bow_to_predict),
            'boi_cru': (boi_treino, boi_to_predict),
            'bow_cru': (bow_resumo_treino, bow_resumo_to_predict),
            'completo': (df_completo_treino, df_completo_to_predict),
        }

        return dataframes
