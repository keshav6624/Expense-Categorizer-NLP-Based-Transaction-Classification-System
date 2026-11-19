import unittest

try:
    from nlp.text_processor import clean_text, extract_features
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    clean_text = None
    extract_features = None
    IMPORT_ERROR = exc


class TextProcessorTests(unittest.TestCase):
    @unittest.skipIf(IMPORT_ERROR is not None, f"Missing dependency: {IMPORT_ERROR}")
    def test_clean_text_normalizes_punctuation_and_case(self):
        self.assertEqual(clean_text(" Starbucks! Coffee?? "), "starbucks coffee")

    @unittest.skipIf(IMPORT_ERROR is not None, f"Missing dependency: {IMPORT_ERROR}")
    def test_extract_features_returns_expected_keys_for_empty_input(self):
        features = extract_features(None)

        self.assertIn("entities", features)
        self.assertIn("keywords", features)
        self.assertIn("sentiment", features)
        self.assertIn("is_refund", features)
        self.assertIsInstance(features["entities"], list)
        self.assertIsInstance(features["keywords"], list)
        self.assertEqual(features["is_refund"], 0)


if __name__ == "__main__":
    unittest.main()