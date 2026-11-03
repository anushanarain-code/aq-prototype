
import unittest

import pandas as pd

from matching import build_match_index, match_holding


DATASET_PATH = "data/aim_companies.csv"


class TestPortfolioMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the real dataset once for the test suite."""
        df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
        cls.index = build_match_index(df)

    def test_bajaj_matches_green_by_isin(self):
        result = match_holding(
            "Bajaj Auto Ltd.", "INE917I01010", "BAJAJ-AUTO", self.index
        )
        self.assertEqual(result["matched_company"], "Bajaj Auto Ltd.")
        self.assertEqual(result["band"], "Green")
        self.assertEqual(result["match_type"], "isin")

    def test_eicher_matches_orange_by_isin(self):
        result = match_holding(
            "Eicher Motors Ltd.", "INE066A01021", "EICHERMOT", self.index
        )
        self.assertEqual(result["band"], "Orange")
        self.assertEqual(result["match_type"], "isin")

    def test_isin_matching_ignores_case_and_surrounding_spaces(self):
        result = match_holding(
            "Bajaj Auto Ltd.", " ine917i01010 ", "", self.index
        )
        self.assertEqual(result["band"], "Green")
        self.assertEqual(result["match_type"], "isin")

    def test_exact_name_match(self):
        result = match_holding(
            "Bajaj Auto Ltd.", "", "", self.index
        )
        self.assertEqual(result["band"], "Green")
        self.assertEqual(result["match_type"], "exact_name")

    def test_name_normalisation_handles_punctuation_and_legal_suffix(self):
        result = match_holding(
            "Eicher Motors, Limited", "", "", self.index
        )
        self.assertEqual(result["band"], "Orange")
        self.assertEqual(result["match_type"], "exact_name")

    def test_partial_name_does_not_receive_a_band(self):
        result = match_holding("Bajaj", "", "", self.index)
        self.assertEqual(result["band"], "")
        self.assertEqual(result["match_type"], "none")

    def test_missing_identifiers_remain_unclassified(self):
        result = match_holding("", "", "", self.index)
        self.assertEqual(result["band"], "")
        self.assertEqual(result["status"], "Missing identifiers")

    def test_unknown_isin_falls_back_to_valid_symbol_with_warning(self):
        result = match_holding(
            "Bajaj Auto Ltd.", "UNKNOWN12345", "BAJAJ-AUTO", self.index
        )
        self.assertEqual(result["band"], "Green")
        self.assertEqual(result["match_type"], "symbol")
        self.assertTrue(result["warnings"])
        self.assertIn("review", result["status"].lower())

    def test_mismatched_name_is_flagged_for_review(self):
        result = match_holding(
            "Eicher Motors Ltd.", "INE917I01010", "BAJAJ-AUTO", self.index
        )
        self.assertEqual(result["matched_company"], "Bajaj Auto Ltd.")
        self.assertEqual(result["band"], "Green")
        self.assertIn("review", result["status"].lower())
        self.assertTrue(result["warnings"])

    def test_duplicate_isin_remains_unclassified(self):
        df = pd.DataFrame([
            {
                "Company Name": "Company A Ltd.",
                "ISIN Code": "INE000A01000",
                "Symbol": "COMPANYA",
                "Category": "Green",
            },
            {
                "Company Name": "Company B Ltd.",
                "ISIN Code": "INE000A01000",
                "Symbol": "COMPANYB",
                "Category": "Orange",
            },
        ])
        index = build_match_index(df)

        result = match_holding(
            "Company A Ltd.",
            "INE000A01000",
            "COMPANYA",
            index,
        )

        self.assertEqual(result["band"], "")
        self.assertIn("review", result["status"].lower())

    def test_duplicate_symbol_remains_unclassified_even_with_exact_name(self):
        df = pd.DataFrame([
            {
                "Company Name": "Company A Ltd.",
                "ISIN Code": "INE000A01001",
                "Symbol": "SHARED",
                "Category": "Green",
            },
            {
                "Company Name": "Company B Ltd.",
                "ISIN Code": "INE000B01001",
                "Symbol": "SHARED",
                "Category": "Orange",
            },
        ])
        index = build_match_index(df)

        result = match_holding(
            "Company A Ltd.",
            "",
            "SHARED",
            index,
        )

        self.assertEqual(result["band"], "")
        self.assertIn("review", result["status"].lower())

    def test_unknown_company_is_not_assessed(self):
        result = match_holding(
            "Imaginary Company Ltd.", "", "", self.index
        )
        self.assertEqual(result["band"], "")
        self.assertEqual(result["match_type"], "none")

    def test_duplicate_normalised_names_remain_unclassified(self):
        # These names normalise to the same key.
        df = pd.DataFrame([
            {
                "Company Name": "Example Ltd.",
                "ISIN Code": "AA123456781",
                "Symbol": "EX1",
                "Category": "Green",
            },
            {
                "Company Name": "Example Limited",
                "ISIN Code": "BB123456782",
                "Symbol": "EX2",
                "Category": "Red",
            },
        ])
        index = build_match_index(df)

        result = match_holding("Example", "", "", index)

        self.assertEqual(result["band"], "")
        self.assertEqual(result["status"], "Ambiguous — needs review")
        self.assertEqual(len(index["collisions"]["name"]), 1)

    def test_duplicate_symbols_are_reported(self):
        df = pd.DataFrame([
            {
                "Company Name": "First Company",
                "ISIN Code": "AA123456781",
                "Symbol": "DUP",
                "Category": "Green",
            },
            {
                "Company Name": "Second Company",
                "ISIN Code": "BB123456782",
                "Symbol": "DUP",
                "Category": "Red",
            },
        ])
        index = build_match_index(df)

        result = match_holding("", "", "DUP", index)

        self.assertEqual(result["band"], "")
        self.assertEqual(len(index["collisions"]["symbol"]), 1)

    def test_invalid_dataset_band_is_not_returned(self):
        df = pd.DataFrame([
            {
                "Company Name": "Test Company",
                "ISIN Code": "AA123456781",
                "Symbol": "TEST",
                "Category": "Purple",
            },
        ])
        index = build_match_index(df)

        result = match_holding("Test Company", "", "", index)

        self.assertEqual(result["band"], "")
        self.assertTrue(result["warnings"])
        self.assertIn("review", result["status"].lower())


if __name__ == "__main__":
    unittest.main()
