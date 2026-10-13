import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from avito_candidates.fitted_fingerprint import fitted_estimator_fingerprint, classifier_semantic_fingerprint


class FittedFingerprintTest(unittest.TestCase):
    def fixture(self):
        documents = ["fix tap service", "repair water pipe", "buy chair furniture", "sell table furniture"]
        word = TfidfVectorizer(ngram_range=(1, 2), dtype=np.float32)
        char = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), dtype=np.float32)
        matrix = sparse.hstack([word.fit_transform(documents), char.fit_transform(documents)], format="csr")
        estimator = SGDClassifier(loss="log_loss", max_iter=2, tol=None, random_state=42).fit(matrix, [0, 0, 1, 1])
        model = SimpleNamespace(word=word, char=char, classifier=estimator,
            classes=estimator.classes_, fitted_texts=frozenset(documents),
            feature_count=matrix.shape[1], training_examples=4, total_weight=4.)
        wrapper = SimpleNamespace(model=model, crossfit_models={0: copy.deepcopy(model)},
                                  metadata={"seed": 42, "classes": [0, 1]}, _prediction_cache={})
        return wrapper, documents

    def predict(self, model, documents):
        matrix = sparse.hstack([model.word.transform(documents), model.char.transform(documents)], format="csr")
        return model.classifier.predict_proba(matrix)

    def test_stable_before_after_prediction_and_compressed_joblib_roundtrip(self):
        wrapper, documents = self.fixture()
        before = classifier_semantic_fingerprint(wrapper)
        expected = self.predict(wrapper.model, documents)
        self.assertEqual(before, classifier_semantic_fingerprint(wrapper))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.joblib"
            joblib.dump(wrapper, path, compress=3)
            restored = joblib.load(path)
            # Simulate the saved pointer from a different Python process. This
            # is not a stop-word vocabulary or learned numerical parameter.
            restored.model.word._stop_words_id = -123
            self.assertEqual(before, classifier_semantic_fingerprint(restored))
            np.testing.assert_array_equal(expected, self.predict(restored.model, documents))
            self.assertEqual(before, classifier_semantic_fingerprint(restored))

    def test_all_meaningful_fitted_signals_and_metadata_remain_bound(self):
        wrapper, _documents = self.fixture()
        before = classifier_semantic_fingerprint(wrapper)
        mutations = [
            lambda x: setattr(x.model, "fitted_texts", x.model.fitted_texts | {"new member"}),
            lambda x: setattr(x.crossfit_models[0], "fitted_texts", frozenset({"other fold"})),
            lambda x: x.model.classifier.coef_.__setitem__((0, 0), x.model.classifier.coef_[0, 0] + .01),
            lambda x: x.model.classifier.intercept_.__setitem__(0, x.model.classifier.intercept_[0] + .01),
            lambda x: setattr(x.model.classifier, "classes_", x.model.classifier.classes_[::-1]),
            lambda x: setattr(x.model.classifier, "coef_", x.model.classifier.coef_.astype(np.float64)),
            lambda x: setattr(x.model.word, "idf_", x.model.word.idf_ + .01),
            lambda x: x.model.word.vocabulary_.__setitem__("new token", 1000),
            lambda x: x.model.word.set_params(ngram_range=(1, 1)),
            lambda x: x.metadata.__setitem__("seed", 13),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = copy.deepcopy(wrapper)
                mutate(changed)
                self.assertNotEqual(before, classifier_semantic_fingerprint(changed))
        wrapper._prediction_cache["disposable"] = [("x", .5)]
        self.assertEqual(before, classifier_semantic_fingerprint(wrapper))

    def test_transient_exception_does_not_drop_same_named_generic_model_field(self):
        model = SimpleNamespace(_stop_words_id=1, fitted_texts=frozenset({"a", "b"}))
        before = fitted_estimator_fingerprint(model)
        model._stop_words_id = 2
        self.assertNotEqual(before, fitted_estimator_fingerprint(model))


if __name__ == "__main__":
    unittest.main()
