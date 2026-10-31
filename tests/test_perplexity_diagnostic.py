"""留出诊断的拆分、fit隔离及检查点测试。"""
from pathlib import Path
import sys, tempfile, unittest
import numpy as np
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_perplexity_diagnostic as diagnostic

class Tests(unittest.TestCase):
    def test_split_fixed_complete_disjoint(self):
        config = {"test_size": .2, "random_state": 20260904, "shuffle": True}
        for n in [43019, 43016, 43015, 43014]:
            a, b = diagnostic.make_split(n, config); c, d = diagnostic.make_split(n, config)
            np.testing.assert_array_equal(a, c); np.testing.assert_array_equal(b, d)
            self.assertEqual(len(b), int(np.ceil(n * .2))); self.assertFalse(set(a) & set(b)); self.assertEqual(len(set(a) | set(b)), n)

    def test_fit_receives_train_only(self):
        train = sparse.csr_matrix([[2, 1, 0], [1, 0, 4], [0, 2, 3]]); test = sparse.csr_matrix([[0, 7, 1]])
        class Spy(LatentDirichletAllocation):
            def fit(self, X, y=None): self.fit_input = X; return super().fit(X, y)
        model = Spy(n_components=2, max_iter=2, random_state=42, n_jobs=1); result = diagnostic.evaluate_one(model, train, test)
        self.assertIs(model.fit_input, train); self.assertGreater(result["test_perplexity"], 0)

    def test_checkpoint_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.json"; result = {"test_perplexity": 12.}
            item = {"checkpoint_signature": "x", "result": result, "result_sha256": diagnostic.signature(result)}; diagnostic.atomic_json(path, item)
            self.assertEqual(diagnostic.load_checkpoint(path, "x"), result)
            item["result"]["test_perplexity"] = 1.; diagnostic.atomic_json(path, item)
            with self.assertRaises(ValueError): diagnostic.load_checkpoint(path, "x")

    def test_factory_parameters(self):
        config = diagnostic.read_json(diagnostic.ROOT / "config/tf_df_lda_config.yaml")["lda"]; model = diagnostic.rebuild.build_model(40, 2026, config)
        for key in ["learning_method", "max_iter", "evaluate_every", "perp_tol", "mean_change_tol", "max_doc_update_iter", "n_jobs", "doc_topic_prior", "topic_word_prior", "verbose"]: self.assertEqual(model.get_params()[key], config[key])
        self.assertEqual((model.n_components, model.random_state), (40, 2026))

    def test_row_order_digest(self): self.assertNotEqual(diagnostic.row_digest([1, 2]), diagnostic.row_digest([2, 1]))

if __name__ == "__main__": unittest.main()
