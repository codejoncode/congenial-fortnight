import pytest
import numpy as np


class DummyForecasting:
    def __init__(self, X_train, y_train, X_val=None, y_val=None):
        self._data = (X_train, y_train, X_val, y_val)

    def load_and_prepare_datasets(self):
        return self._data


def dummy_trainer_ok(X_train, y_train, X_val, y_val, pair_name=None):
    # simulate a successful training run
    class Model:
        def __init__(self):
            self._trained = True

        def num_trees(self):
            return 10

    return Model()


def test_missing_training_data(monkeypatch, tmp_path):
    # Missing X_train / y_train should be rejected
    from scripts.automated_training import AutomatedTrainer

    trainer = AutomatedTrainer()

    monkeypatch.setattr('scripts.automated_training.ForecastingSystem', lambda pair: DummyForecasting(None, None))
    res = trainer.run_automated_training(['TESTPAIR'])
    assert 'TESTPAIR' in res
    assert 'Missing training data' in res['TESTPAIR']['error'] or 'Missing training data or labels' in res['TESTPAIR']['error']


def test_mismatched_lengths(monkeypatch):
    from scripts.automated_training import AutomatedTrainer
    trainer = AutomatedTrainer()

    X = np.zeros((100, 5))
    y = np.zeros(50)  # intentionally mismatched

    monkeypatch.setattr('scripts.automated_training.ForecastingSystem', lambda pair: DummyForecasting(X, y))
    res = trainer.run_automated_training(['PAIR2'])
    assert 'PAIR2' in res
    assert 'Mismatched lengths' in res['PAIR2']['error']


def test_empty_arrays(monkeypatch):
    from scripts.automated_training import AutomatedTrainer
    trainer = AutomatedTrainer()

    X = np.zeros((0, 5))
    y = np.zeros((0,))

    monkeypatch.setattr('scripts.automated_training.ForecastingSystem', lambda pair: DummyForecasting(X, y))
    res = trainer.run_automated_training(['EMPTY'])
    assert 'EMPTY' in res
    assert 'Empty training arrays' in res['EMPTY']['error']


def test_small_dataset_uses_emergency_config(monkeypatch):
    from scripts.automated_training import AutomatedTrainer
    trainer = AutomatedTrainer()

    X = np.zeros((8, 3))
    y = np.zeros((8,))

    monkeypatch.setattr('scripts.automated_training.ForecastingSystem', lambda pair: DummyForecasting(X, y))
    # Replace training pipeline with dummy that returns a model
    monkeypatch.setattr('scripts.automated_training.enhanced_lightgbm_training_pipeline', dummy_trainer_ok)

    res = trainer.run_automated_training(['SMALL'])
    assert 'SMALL' in res
    assert res['SMALL']['status'] == 'success'
