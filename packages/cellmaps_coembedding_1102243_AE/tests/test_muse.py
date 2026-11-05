import csv
import os
import shutil
import tempfile
import unittest

import numpy as np

from cellmaps_coembedding.muse_sc import make_matrix_from_labels, write_result_to_file


class TestMuse(unittest.TestCase):
    def test_make_matrix_from_labels(self):
        labels = np.array([0, 0, 1, 1, 2])
        expected_matrix = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        result_matrix = make_matrix_from_labels(labels)
        np.testing.assert_array_equal(result_matrix, expected_matrix)

    def test_single_cluster(self):
        labels = np.array([0, 0, 0])
        expected = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        result = make_matrix_from_labels(labels)
        np.testing.assert_array_equal(result, expected)

    def test_multiple_clusters(self):
        labels = np.array([0, 1, 0, 1, 2])
        expected = np.array([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        result = make_matrix_from_labels(labels)
        np.testing.assert_array_equal(result, expected)

    def test_write_result_to_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(temp_dir, "results.tsv")
            data = np.array([[1, 2], [3, 4]])
            indexes = [0, 1]
            write_result_to_file(filepath, data, indexes)
            expected_rows = [
                ['', '1'],
                ['0', '1', '2'],
                ['1', '3', '4']
            ]
            with open(filepath, 'r', newline='') as file:
                reader = csv.reader(file, delimiter='\t')
                for i, row in enumerate(reader):
                    self.assertEqual(row, expected_rows[i])
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
