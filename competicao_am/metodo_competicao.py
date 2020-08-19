from base_am.metodo import MetodoAprendizadoDeMaquina
import pandas as pd
from .preprocessamento_atributos_competicao import gerar_atributos_ator, gerar_atributos_resumo
from base_am.resultado import Resultado
from typing import Union, List
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC


class MetodoCompeticao(MetodoAprendizadoDeMaquina):
    def __init__(self, ml_method: Union[ClassifierMixin, RegressorMixin]):
        self.ml_method = ml_method

        self.dic_int_to_nom_classe = {}
        self.dic_nom_classe_to_int = {}

    def class_to_number(self, y):
        arr_int_y = []

        for rotulo_classe in y:
            if rotulo_classe not in self.dic_nom_classe_to_int:
                int_new_val_classe = len(self.dic_nom_classe_to_int.keys())
                self.dic_nom_classe_to_int[rotulo_classe] = int_new_val_classe
                self.dic_int_to_nom_classe[int_new_val_classe] = rotulo_classe

            arr_int_y.append(self.dic_nom_classe_to_int[rotulo_classe])

        return arr_int_y

    def obtem_y(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        y_treino = self.class_to_number(df_treino[col_classe])
        y_to_predict = None

        if col_classe in df_data_to_predict.columns:
            y_to_predict = self.class_to_number(df_data_to_predict[col_classe])

        return y_treino, y_to_predict

    def obtem_x(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str):
        x_treino = df_treino.drop(col_classe, axis=1)
        x_to_predict = df_data_to_predict
        if col_classe in df_data_to_predict.columns:
            x_to_predict = df_data_to_predict.drop(col_classe, axis=1)
        return x_treino, x_to_predict

    def eval_actors(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str, seed: int = 1):
        x_treino, x_to_predict = self.obtem_x(
            df_treino, df_data_to_predict, col_classe)
        y_treino, y_to_predict = self.obtem_y(
            df_treino, df_data_to_predict, col_classe)

        df_treino_ator, df_to_predict_ator = gerar_atributos_ator(
            x_treino, x_to_predict)

        arr_df_to_remove_id = [df_treino_ator, df_to_predict_ator]
        for df_data in arr_df_to_remove_id:
            df_data.drop("id", axis=1)

        self.ml_method.fit(df_treino_ator, y_treino)
        arr_predict = self.ml_method.predict(df_to_predict_ator)

        return y_to_predict, arr_predict

    def eval_bow(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str, seed: int = 1):
        x_treino, x_to_predict = self.obtem_x(
            df_treino, df_data_to_predict, col_classe)
        y_treino, y_to_predict = self.obtem_y(
            df_treino, df_data_to_predict, col_classe)

        df_treino_bow, df_to_predict_bow = gerar_atributos_resumo(
            x_treino, x_to_predict)

        self.ml_method.fit(df_treino_bow, y_treino)
        arr_predict = self.ml_method.predict(df_to_predict_bow)

        return y_to_predict, arr_predict

    def combine_predictions(self, arr_predictions_1: List[int], arr_predictions_2: List[int]) -> List[int]:

        y_final_predictions = []
        for i, pred in enumerate(arr_predictions_1):
            if self.dic_int_to_nom_classe[pred] == 'Comedy':
                y_final_predictions.append(pred)
            else:
                y_final_predictions.append(arr_predictions_2[i])
        return y_final_predictions

    def eval(self, df_treino: pd.DataFrame, df_data_to_predict: pd.DataFrame, col_classe: str, seed: int = 1):

        y_to_predict, arr_predictions_ator = self.eval_actors(
            df_treino, df_data_to_predict, col_classe)

        y_to_predict, arr_predictions_bow = self.eval_bow(
            df_treino, df_data_to_predict, col_classe)

        arr_final_predictions = self.combine_predictions(
            arr_predictions_ator, arr_predictions_bow)

        return Resultado(y_to_predict, arr_final_predictions)
