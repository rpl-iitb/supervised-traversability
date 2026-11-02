import unittest

from learning.real_data import run_folds, train_val_test


def table():
    runs = {}
    for i in range(30):
        runs[f"h{i}"] = {"condition": "hazard", "session": "s1c" if i % 3 else "s1a", "direction": "A->B" if i % 2 else "B->A"}
    for i in range(6):
        runs[f"c{i}"] = {"condition": "clean", "session": "s1c" if i % 2 else "s1a", "direction": "A->B" if i % 3 else "B->A"}
    return runs


class SplitTest(unittest.TestCase):
    def test_no_run_is_in_two_splits_and_every_run_is_tested_once(self):
        runs = table()
        folds = run_folds(runs, 5)
        tested = []
        for i in range(5):
            train, val, test = train_val_test(runs, folds, i)
            self.assertFalse(train & val or train & test or val & test)
            self.assertEqual(train | val | test, set(runs))
            tested.extend(test)
        self.assertEqual(sorted(tested), sorted(runs))

    def test_every_fold_gets_a_clean_run_when_there_are_enough(self):
        for fold in run_folds(table(), 5):
            self.assertTrue(any(name.startswith("c") for name in fold))

    def test_same_seed_same_folds(self):
        self.assertEqual(run_folds(table(), 5, seed=3), run_folds(table(), 5, seed=3))


if __name__ == "__main__":
    unittest.main()
