"""
src/models/regime_detection.py — Detección de regímenes con HMM y VaR condicional.

Proporciona clases para entrenar un HMM sobre retornos (y variables adicionales),
predecir regímenes, calcular probabilidades de estado y generar VaR régimen-ajustado
para comparar con el VaR incondicional.
"""

import json
from collections import namedtuple
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logger.warning('hmmlearn no disponible — HMM de régimen no funcionará')


RegimeInfo = namedtuple('RegimeInfo', ['name', 'multiplier', 'description'])
REGIMES = [
    RegimeInfo('Calma', 0.6, 'Bajo ruido y volatilidad, mercado en calma'),
    RegimeInfo('Normal', 1.0, 'Condiciones estándar de mercado'),
    RegimeInfo('Bear/High Vol', 1.8, 'Mercado con volatilidad elevada y retornos negativos'),
    RegimeInfo('Crisis/Tail', 3.5, 'Evento extremo con cola gruesa y crisis financiera'),
]


class HMMRegimeDetector:
    """Detector de regímenes usando un HMM gaussiano."""

    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: str = 'full',
        n_iter: int = 200,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.fitted = False

    def _build_feature_matrix(
        self,
        returns: pd.Series,
        additional: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        X = returns.to_numpy(dtype=float).reshape(-1, 1)
        if additional is not None:
            additional = additional.reindex(returns.index).ffill().fillna(0)
            if not additional.empty:
                X = np.hstack([X, additional.to_numpy(dtype=float)])
        return X

    def fit(
        self,
        returns: pd.Series,
        additional: Optional[pd.DataFrame] = None,
    ):
        if not HMMLEARN_AVAILABLE:
            raise ImportError('hmmlearn no está instalado; instale hmmlearn para usar HMMRegimeDetector')

        X = self._build_feature_matrix(returns, additional)
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
            min_covar=1e-3,
        )
        self.model.fit(X)
        self.fitted = True
        logger.info(f'HMM entrenado: {self.n_regimes} regímenes, cov={self.covariance_type}, iter={self.n_iter}')
        return self

    def predict_regimes(
        self,
        returns: pd.Series,
        additional: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        if not self.fitted or self.model is None:
            raise RuntimeError('HMM no entrenado. Ejecute detector.fit(...) primero.')
        X = self._build_feature_matrix(returns, additional)
        return self.model.predict(X)

    def predict_regime_probs(
        self,
        returns: pd.Series,
        additional: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if not self.fitted or self.model is None:
            raise RuntimeError('HMM no entrenado. Ejecute detector.fit(...) primero.')
        X = self._build_feature_matrix(returns, additional)
        probs = self.model.predict_proba(X)
        cols = [f'regime_prob_{i}' for i in range(self.n_regimes)]
        return pd.DataFrame(probs, index=returns.index, columns=cols)

    def get_transition_matrix(self) -> pd.DataFrame:
        if not self.fitted or self.model is None:
            return pd.DataFrame()
        labels = [f'regime_{i}' for i in range(self.n_regimes)]
        return pd.DataFrame(self.model.transmat_, index=labels, columns=labels)

    def save(self, path: str):
        if not self.fitted or self.model is None:
            raise RuntimeError('HMM no entrenado. No hay modelo para guardar.')
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'n_regimes': self.n_regimes,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter,
            'random_state': self.random_state,
            'startprob': self.model.startprob_.tolist(),
            'transmat': self.model.transmat_.tolist(),
            'means': self.model.means_.tolist(),
            'covars': self.model.covars_.tolist(),
        }
        with output.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        logger.info('Modelo HMM guardado en %s', output)


class RegimeAwareVaR:
    """Calcula VaR condicional al régimen usando la inferencia de HMM."""

    def __init__(
        self,
        detector: HMMRegimeDetector,
        baseline_quantile: float = 0.01,
        multipliers: Optional[Dict[int, float]] = None,
    ):
        self.detector = detector
        self.baseline_quantile = baseline_quantile
        self.multipliers = multipliers or {i: REGIMES[i].multiplier if i < len(REGIMES) else 1.0 for i in range(detector.n_regimes)}

    def calculate_regime_var(
        self,
        returns: pd.Series,
        additional: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        baseline = float(returns.quantile(self.baseline_quantile))
        regimes = self.detector.predict_regimes(returns, additional)
        regime_probs = self.detector.predict_regime_probs(returns, additional)

        multipliers = np.array([self.multipliers.get(int(r), 1.0) for r in regimes])
        var_regime = baseline * multipliers

        if regime_probs.shape[1] == len(self.multipliers):
            weighted_multipliers = regime_probs.dot(np.array([self.multipliers.get(i, 1.0) for i in range(regime_probs.shape[1])]))
        else:
            weighted_multipliers = np.full(len(regime_probs), 1.0)

        var_prob_weighted = baseline * weighted_multipliers

        df = pd.DataFrame(
            {
                'current_regime': regimes,
                'var_unconditional': baseline,
                'var_regime_adjusted': var_regime,
                'var_prob_weighted': var_prob_weighted,
            },
            index=returns.index,
        )
        return df

    def compare_backtesting(
        self,
        returns: pd.Series,
        var_df: pd.DataFrame,
        window: int = 250,
    ) -> Dict[str, Dict]:
        tail = returns.index[-window:] if len(returns) >= window else returns.index
        returns_tail = returns.loc[tail]
        var_tail = var_df.reindex(tail)

        results = {}
        for label in ['var_unconditional', 'var_regime_adjusted', 'var_prob_weighted']:
            if label not in var_tail.columns:
                continue
            exceedances = returns_tail[returns_tail < var_tail[label]].count()
            rate = exceedances / len(var_tail)
            if rate <= 0.01:
                traffic = 'green'
            elif rate <= 0.02:
                traffic = 'yellow'
            else:
                traffic = 'red'
            results[label] = {
                'n_exceedances': int(exceedances),
                'exceedance_rate': float(rate),
                'traffic_light': traffic,
            }

        if 'var_unconditional' in results and 'var_regime_adjusted' in results:
            results['var_regime_adjusted']['improvement_vs_baseline'] = (
                f"{results['var_unconditional']['n_exceedances'] - results['var_regime_adjusted']['n_exceedances']} exceedances menos"
            )
        if 'var_unconditional' in results and 'var_prob_weighted' in results:
            results['var_prob_weighted']['improvement_vs_baseline'] = (
                f"{results['var_unconditional']['n_exceedances'] - results['var_prob_weighted']['n_exceedances']} exceedances menos"
            )

        return results
