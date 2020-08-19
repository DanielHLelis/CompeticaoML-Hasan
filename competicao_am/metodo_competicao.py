from base_am.metodo import MetodoAprendizadoDeMaquina
import pandas as pd
from .preprocessamento_atributos_competicao import DataframePreprocessing
from base_am.resultado import Resultado
from typing import Union, List, Tuple
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from IPython.display import display

# TODO:
# [] Implement multi ml_method support;
# [] Verify unused properties;


class MetodoCompeticao(MetodoAprendizadoDeMaquina):
    def __init__(self, ml_method: Union[ClassifierMixin, RegressorMixin], preprocessor: DataframePreprocessing = None, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None):
        self.ml_method = ml_method
        self.preprocessor = preprocessor

        if df_treino is not None and df_data_to_predict is not None and col_classe is not None and preprocessor is None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

    # Properties

    #! Might remove

    @property
    def df_treino(self):
        return self.preprocessor.df_treino

    @property
    def df_data_to_predict(self):
        return self.preprocessor.df_data_to_predict

    @property
    def df_data_to_predict(self):
        return self.preprocessor.col_classe

    @property
    def x(self):
        return self.preprocessor.x

    @property
    def y(self):
        return self.preprocessor.y

    #! End

    # Combiner

    def combine_predictions(self, *predictions: Tuple[List[int]]) -> List[int]:
        y_final_predictions = []

        if len(predictions) < 1:
            raise ValueError("Predictins can't be empty")

        for i in range(len(predictions[0])):
            comedy = 0
            action = 0

            for prediction in predictions:
                if self.preprocessor.nom_classe(prediction[i]) == 'Comedy':
                    comedy += 1
                else:
                    action += 1

            if comedy > action:
                y_final_predictions.append(
                    self.preprocessor.int_classe('Comedy'))
            elif comedy == action:
                y_final_predictions.append(predictions[0][i])
            else:
                y_final_predictions.append(
                    self.preprocessor.int_classe('Action'))

        return y_final_predictions

    # Evals

    def _eval_boi(self, items: List[str], min_occur: int = 3, seed: int = 1):
        y_treino, y_to_predict = self.y

        df_treino_boi, df_to_predict_boi = self.preprocessor.bag_of_items(
            items, min_occur)
        df_treino_boi, df_to_predict_boi = DataframePreprocessing.remove_id(
            df_treino_boi, df_to_predict_boi)

        # ? Should the seed go here?
        self.ml_method.fit(df_treino_boi, y_treino)
        arr_predict = self.ml_method.predict(df_to_predict_boi)

        return y_to_predict, arr_predict

    def _eval_bow(self, column: str, seed: int = 1):
        y_treino, y_to_predict = self.y

        df_treino_bow, df_to_predict_bow = self.preprocessor.bag_of_words(
            column)

        # df_treino_bow, df_to_predict_bow = DataframePreprocessing.remove_id(
        #     df_treino_bow, df_to_predict_bow)

        # ? Should the seed go here?
        self.ml_method.fit(df_treino_bow, y_treino)
        arr_predict = self.ml_method.predict(df_to_predict_bow)

        return y_to_predict, arr_predict

    def eval(self, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None, seed: int = 1) -> Resultado:
        if df_treino is not None and df_data_to_predict is not None and col_classe is not None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

        y_to_predict, predictions_ator = self._eval_boi(
            ["ator_1", "ator_2", "ator_3", "ator_4", "ator_5"])
        y_to_predict, predictions_escritor = self._eval_boi(
            ["escrito_por_1", "escrito_por_2"])  # Novo
        y_to_predict, predictions_resumo = self._eval_bow("resumo")
        y_to_predict, predictions_titulo = self._eval_bow("titulo")  # Novo

        final_predictions = self.combine_predictions(
            predictions_resumo, predictions_titulo, predictions_escritor, predictions_ator)  # Ordem indica prioridade

        return Resultado(y_to_predict, final_predictions)
