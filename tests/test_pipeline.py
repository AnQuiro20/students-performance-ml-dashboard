import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml import train_model
from utils import load_data


class PipelineTests(unittest.TestCase):
    def test_load_data_reads_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            pd.DataFrame({"math score": [80]}).to_csv(path, index=False)

            result = load_data(path, logging.getLogger("test"))

        self.assertEqual(result.to_dict(orient="records"), [{"math score": 80}])

    def test_training_pipeline_returns_predictions(self):
        rows = 30
        data = pd.DataFrame(
            {
                "gender": ["female", "male"] * (rows // 2),
                "parental level of education": ["bachelor's degree", "high school"] * (rows // 2),
                "lunch": ["standard", "free/reduced"] * (rows // 2),
                "test preparation course": ["completed", "none"] * (rows // 2),
                "reading score": list(range(55, 85)),
                "writing score": list(range(50, 80)),
                "math score": list(range(52, 82)),
            }
        )

        model, name, results, y_test, predictions, _, _ = train_model(
            data, logging.getLogger("test")
        )

        self.assertIsNotNone(model)
        self.assertIn(name, {"Linear Regression", "Random Forest"})
        self.assertEqual(set(results), {"Linear Regression", "Random Forest"})
        self.assertEqual(len(predictions), len(y_test))


if __name__ == "__main__":
    unittest.main()
