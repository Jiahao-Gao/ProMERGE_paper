import shutil
import unittest
import tempfile
from unittest.mock import patch

from cellmaps_coembedding.runner import FakeCoEmbeddingGenerator


class TestFakeEmbeddingGenerator(unittest.TestCase):
    """Tests for `EmbeddingGenerator` class."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Tear down test fixtures, if any."""
        shutil.rmtree(self.temp_dir)

    @patch('random.random')
    @patch.object(FakeCoEmbeddingGenerator, '_get_embeddings_and_names')
    def test_get_next_embedding(self, mock_get_embeddings_and_names, mock_random):
        generator_fake = FakeCoEmbeddingGenerator(dimensions=128, ppi_embeddingdir=self.temp_dir,
                                                  image_embeddingdir=self.temp_dir)

        def mock_embeddings_and_names():
            return [[['gene1', 0.1, 0.2], ['gene2', 0.3, 0.4]], 
                   [['gene2', 0.5, 0.6], ['gene3', 0.7, 0.8]]], ['PPI', 'image']

        
        mock_get_embeddings_and_names.side_effect = mock_embeddings_and_names

        mock_random.return_value = 0.5

        result_embeddings = list(generator_fake.get_next_embedding())

        self.assertEqual(len(result_embeddings), 1)
        for embed in result_embeddings:
            self.assertEqual(len(embed) - 1, 128)
            self.assertEqual(embed[0], 'gene2')
