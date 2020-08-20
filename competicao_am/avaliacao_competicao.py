from base_am.avaliacao import OtimizacaoObjetivo
from base_am.metodo import MetodoAprendizadoDeMaquina
from base_am.resultado import Fold, Resultado
import competicao_am.metodo_competicao as mc
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

        return mc.MetodoCompeticao(scikit_method)

    def resultado_metrica_otimizacao(self, resultado: Resultado) -> float:
        return resultado.macro_f1


class OtimizacaoObjetivoSVCRbf(OtimizacaoObjetivo):
    def __init__(self, fold: Fold, metodo: mc.MetodoCompeticao.__class__):
        super().__init__(fold)
        self.metodo = metodo

    def obtem_metodo(self, trial: optuna.Trial):
        exp_cost = trial.suggest_uniform('exp_cost', 0, 7)

        C = 2 ** exp_cost

        scikit_method = svm.SVC(
            C=C, kernel='rbf', random_state=2)

        return self.metodo(scikit_method)

    def resultado_metrica_otimizacao(self, resultado: Resultado) -> float:
        return resultado.macro_f1


class OtimizacaoObjetivoSVCRbf(OtimizacaoObjetivo):
    def __init__(self, fold: Fold, metodo: mc.MetodoCompeticao.__class__ = mc.MetodoCompeticao):
        super().__init__(fold)
        self.metodo = metodo

    def obtem_metodo(self, trial: optuna.Trial):
        exp_cost = trial.suggest_uniform('exp_cost', 0, 7)

        C = 2 ** exp_cost

        scikit_method = svm.SVC(
            C=C, kernel='rbf', random_state=2)

        return self.metodo(scikit_method)

    def resultado_metrica_otimizacao(self, resultado: Resultado) -> float:
        return resultado.macro_f1


class OtimizacaoObjetivoRandomForest(OtimizacaoObjetivo):
    def __init__(self, fold: Fold, num_arvores_max: int = 5, metodo: mc.MetodoCompeticao.__class__ = mc.MetodoCompeticao):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max
        self.metodo = metodo

    def obtem_metodo(self, trial: optuna.Trial) -> MetodoAprendizadoDeMaquina:
        min_samples = trial.suggest_uniform('min_samples_split', 0, 0.5)
        max_features = trial.suggest_uniform('max_features', 0, 0.5)
        num_arvores = trial.suggest_int('num_arvores', 1, self.num_arvores_max)

        clf_rf = RandomForestClassifier(
            min_samples_split=min_samples, max_features=max_features, n_estimators=num_arvores, random_state=2)

        return self.metodo(clf_rf)

    def resultado_metrica_otimizacao(self, resultado: Resultado) -> float:
        return resultado.macro_f1


class WrapperOtimizacaoObjetivo:
    def __init__(self, otimizador: OtimizacaoObjetivo.__class__, **kwargs):
        self.kwargs = kwargs
        self.otimizador = otimizador

    def __call__(self, fold: Fold):
        return self.otimizador(fold=fold, **self.kwargs)
