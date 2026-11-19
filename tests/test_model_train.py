import os
import tempfile
import unittest
from pathlib import Path

try:
    from nlp.model_train import train_model
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    train_model = None
    IMPORT_ERROR = exc


class ModelTrainTests(unittest.TestCase):
    @unittest.skipIf(IMPORT_ERROR is not None, f"Missing dependency: {IMPORT_ERROR}")
    def test_train_model_raises_for_missing_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "bad_training_data.csv"
            csv_path.write_text("description_only\nCoffee\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                train_model(str(csv_path))

    @unittest.skipIf(IMPORT_ERROR is not None, f"Missing dependency: {IMPORT_ERROR}")
    def test_train_model_creates_model_for_small_valid_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "training_data.csv"
            csv_path.write_text(
                "description,category\n"
                "Starbucks coffee,Food\n"
                "Uber ride,Transport\n"
                "Burger meal,Food\n"
                "Taxi trip,Transport\n",
                encoding="utf-8",
            )

            current_dir = os.getcwd()
            os.chdir(temp_dir)
            try:
                model = train_model(str(csv_path))
                self.assertIsNotNone(model)
                self.assertTrue((Path(temp_dir) / "ml_model" / "classifier.pkl").exists())
            finally:
                os.chdir(current_dir)


if __name__ == "__main__":
    unittest.main()