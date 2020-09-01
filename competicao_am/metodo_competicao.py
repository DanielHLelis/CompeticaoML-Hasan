from base_am.metodo import MetodoAprendizadoDeMaquina
from abc import abstractmethod, ABCMeta
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


class MetodoCompeticao(MetodoAprendizadoDeMaquina, metaclass=ABCMeta):
    # def __init__(self, ml_method: Union[ClassifierMixin, RegressorMixin], preprocessor: DataframePreprocessing = None, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None):
    #     self.ml_method = ml_method
    #     self.preprocessor = preprocessor

    #     if df_treino is not None and df_data_to_predict is not None and col_classe is not None and preprocessor is None:
    #         self.preprocessor = DataframePreprocessing(
    #             df_treino, df_data_to_predict, col_classe)

    @property
    def df_treino(self):
        return self.preprocessor.df_treino

    @df_treino.setter
    def df_treino(self, x):
        self.preprocessor.df_treino = x

    @property
    def df_data_to_predict(self):
        return self.preprocessor.df_data_to_predict

    @df_data_to_predict.setter
    def df_data_to_predict(self, x):
        self.preprocessor.df_data_to_predict = x

    @property
    def col_classe(self):
        return self.preprocessor.col_classe

    @property
    def x(self):
        return self.preprocessor.x

    @property
    def y(self):
        return self.preprocessor.y


class MetodoCompeticaoValidacaoSVM(MetodoCompeticao):
    def __init__(self, ml_method: Union[ClassifierMixin, RegressorMixin], preprocessor: DataframePreprocessing = None, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None):
        self.ml_method = ml_method
        self.preprocessor = preprocessor

        if df_treino is not None and df_data_to_predict is not None and col_classe is not None and preprocessor is None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

    def eval(self, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None, seed: int = 1) -> Resultado:
        if df_treino is not None and df_data_to_predict is not None and col_classe is not None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

        # Dataframes

        x_treino, x_to_predict = self.preprocessor.df_resumo()

        y_treino, y_to_predict = self.y

        self.ml_method.fit(x_treino, y_treino)
        predicted = self.ml_method.predict(x_to_predict)

        return Resultado(y_to_predict, predicted)


class MetodoCompeticaoValidacaoRF(MetodoCompeticao):
    def __init__(self, ml_method: Union[ClassifierMixin, RegressorMixin], preprocessor: DataframePreprocessing = None, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None):
        self.ml_method = ml_method
        self.preprocessor = preprocessor

        if df_treino is not None and df_data_to_predict is not None and col_classe is not None and preprocessor is None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

    def eval(self, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None, seed: int = 1) -> Resultado:
        if df_treino is not None and df_data_to_predict is not None and col_classe is not None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

        # Dataframes

        x_treino, x_to_predict = self.preprocessor.df_stats()

        y_treino, y_to_predict = self.y

        self.ml_method.fit(x_treino, y_treino)
        predicted = self.ml_method.predict(x_to_predict)

        return Resultado(y_to_predict, predicted)


class MetodoCompeticaoValidacaoComedy(MetodoCompeticao):
    def __init__(self, ml_method: Union[ClassifierMixin, RegressorMixin], preprocessor: DataframePreprocessing = None, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None):
        self.ml_method = ml_method
        self.preprocessor = preprocessor

        if df_treino is not None and df_data_to_predict is not None and col_classe is not None and preprocessor is None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

    def eval(self, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None, seed: int = 1) -> Resultado:
        if df_treino is not None and df_data_to_predict is not None and col_classe is not None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

        # Dataframes

        x_treino, x_to_predict = self.preprocessor.df_stats()
        y_treino, y_to_predict = self.y

        predicted = [self.preprocessor.int_classe(
            'Comedy')] * len(x_to_predict)

        return Resultado(y_to_predict, predicted)


class MetodoCompeticaoFinal(MetodoCompeticao):
    def __init__(self, svm_method: Union[ClassifierMixin, RegressorMixin], rf_method: Union[ClassifierMixin, RegressorMixin], preprocessor: DataframePreprocessing = None):
        self.svm_method = svm_method
        self.rf_method = rf_method
        self.preprocessor = preprocessor

    def rf_indexes(self) -> List[int]:
        _, x_predict = self.x
        x_predict = x_predict.reset_index()
        return x_predict[x_predict.resumo.apply(self.preprocessor.cleaner).isna()].index.values

    def eval(self, seed: int = 1) -> Resultado:
        # Dataframes
        x_treino_svm, x_to_predict_svm = self.preprocessor.df_resumo()
        x_treino_rf, x_to_predict_rf = self.preprocessor.df_stats()

        y_treino, y_to_predict = self.y

        self.svm_method.fit(x_treino_svm, y_treino)
        predicted_svm = self.svm_method.predict(x_to_predict_svm)

        self.rf_method.fit(x_treino_rf, y_treino)
        predicted_rf = self.rf_method.predict(x_to_predict_rf)

        rf_index = self.rf_indexes()

        final_predictions = [sp if i not in rf_index else predicted_rf[i]
                             for i, sp in enumerate(predicted_svm)]

        return Resultado(y_to_predict, final_predictions)
