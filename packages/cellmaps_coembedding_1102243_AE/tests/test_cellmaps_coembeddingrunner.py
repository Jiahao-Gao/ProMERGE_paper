import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from cellmaps_coembedding.runner import CellmapsCoEmbedder
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError


class TestCellmapsCoEmbedder(unittest.TestCase):

    @patch('cellmaps_coembedding.runner.ProvenanceUtil')
    def test_update_provenance_fields_with_no_rocrate(self, mock_prov_util):
        temp_dir = tempfile.mkdtemp()
        try:
            embedder = CellmapsCoEmbedder(outdir=temp_dir, inputdirs=[temp_dir])
            embedder._update_provenance_fields()
            self.assertEqual(embedder._name, 'Coembedding tool')
        finally:
            shutil.rmtree(temp_dir)

    @patch('cellmaps_coembedding.runner.CellmapsCoEmbedder._register_computation')
    @patch('cellmaps_coembedding.runner.CellmapsCoEmbedder._register_image_coembedding_file')
    @patch('cellmaps_coembedding.runner.CellmapsCoEmbedder._create_rocrate')
    @patch('cellmaps_coembedding.runner.CellmapsCoEmbedder._update_provenance_fields')
    @patch('cellmaps_coembedding.runner.CellmapsCoEmbedder._write_task_start_json')
    def test_run(self, mock_start_json, mock_update_prov, mock_create_rocrate, mock_register_file, mock_register_comp):
        temp_dir = tempfile.mkdtemp()
        try:
            embedder = CellmapsCoEmbedder(outdir=temp_dir, inputdirs=[temp_dir], embedding_generator=MagicMock())
            embedder._description = "A default description"
            embedder._keywords = []
            embedder.run()
            mock_start_json.assert_called_once()
            mock_update_prov.assert_called_once()
            mock_create_rocrate.assert_called_once()
            mock_register_file.assert_called_once()
            mock_register_comp.assert_called_once()
        finally:
            shutil.rmtree(temp_dir)

    def test_constructor_raises_exception_with_no_outdir(self):
        with self.assertRaises(CellmapsCoEmbeddingError):
            CellmapsCoEmbedder()

    def test_run_raises_exception_with_no_outdir(self):
        with self.assertRaises(CellmapsCoEmbeddingError):
            ce = CellmapsCoEmbedder(outdir='abc')
            ce._outdir = None
            ce.run()


if __name__ == '__main__':
    unittest.main()
