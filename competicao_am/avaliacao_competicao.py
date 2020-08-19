from base_am.avaliacao import OtimizacaoObjetivo
from base_am.metodo import MetodoAprendizadoDeMaquina
from base_am.resultado import Fold, Resultado
from competicao_am.metodo_competicao import MetodoCompeticao
import optuna
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


class OtimizacaoObjetivoSVMCompeticao(OtimizacaoObjetivo):
    def __init__(self, fold: Fold, num_arvores_max: int = 5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self, trial: optuna.Trial) -> MetodoAprendizadoDeMaquina:
        # Um custo adequado para custo pode variar muito, por ex, para uma tarefa
        # o valor de custo pode ser 10, para outra, 32000.
        # Assim, normalmente, para conseguir valores mais distintos,
        # usamos c=2^exp_cost
        exp_cost = trial.suggest_uniform('min_samples_split', 0, 7)

        scikit_method = LinearSVC(C=2**exp_cost, random_state=2)

        return MetodoCompeticao(scikit_method)

    def resultado_metrica_otimizacao(self, resultado: Resultado) -> float:
        return resultado.macro_f1


class OtimizacaoObjetivoSVCPolyComGamma(OtimizacaoObjetivo):
    def __init__(self, fold: Fold, degree: int = 2):
        super().__init__(fold)
        self.degree = degree

    def obtem_metodo(self, trial: optuna.Trial):
        exp_cost = trial.suggest_uniform('exp_cost', 0, 7)
        gamma = trial.suggest_uniform('gamma', 0.1, 10)

        C = 2 ** exp_cost

        scikit_method = svm.SVC(
            C=C, gamma=gamma, degree=self.degree, kernel='poly', random_state=2)

        return MetodoCompeticao(scikit_method)

    def resultado_metrica_otimizacao(self, resultado: Resultado) -> float:
        return resultado.macro_f1


class OtimizacaoObjetivoSVCPolySemGamma(OtimizacaoObjetivo):
    def __init__(self, fold: Fold, degree: int = 2):
        super().__init__(fold)
        self.degree = degree

    def obtem_metodo(self, trial: optuna.Trial):
        exp_cost = trial.suggest_uniform('exp_cost', 0, 7)

        C = 2 ** exp_cost

        scikit_method = svm.SVC(
            C=C, degree=self.degree, kernel='poly', random_state=2)

        return MetodoCompeticao(scikit_method)

    def resultado_metrica_otimizacao(self, resultado: Resultado) -> float:
        return resultado.macro_f1


class OtimizacaoObjetivoSVCRbfSemGamma(OtimizacaoObjetivo):
    def __init__(self, fold: Fold):
        super().__init__(fold)

    def obtem_metodo(self, trial: optuna.Trial):
        exp_cost = trial.suggest_uniform('exp_cost', 0, 7)

        C = 2 ** exp_cost

        scikit_method = svm.SVC(
            C=C, kernel='rbf', random_state=2)

        return MetodoCompeticao(scikit_method)

    def resultado_metrica_otimizacao(self, resultado: Resultado) -> float:
        return resultado.macro_f1


class OtimizacaoObjetivoSVCRbfComGamma(OtimizacaoObjetivo):
    def __init__(self, fold: Fold):
        super().__init__(fold)

    def obtem_metodo(self, trial: optuna.Trial):
        exp_cost = trial.suggest_uniform('exp_cost', 0, 7)
        # Talvez o valor seja muito alto
        gamma = trial.suggest_uniform('gamma', 0.1, 10)

        C = 2 ** exp_cost

        scikit_method = svm.SVC(
            C=C, gamma=gamma, kernel='rbf', random_state=2)

        return MetodoCompeticao(scikit_method)

    def resultado_metrica_otimizacao(self, resultado: Resultado) -> float:
        return resultado.macro_f1
