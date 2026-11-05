#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_coembedding` package."""

import os
import shutil
import unittest
import tempfile
from unittest.mock import MagicMock

from cellmaps_utils.exceptions import CellMapsProvenanceError

from cellmaps_coembedding.runner import CellmapsCoEmbedder
from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError


class TestCellmapsCoEmbeddingRunner(unittest.TestCase):
    """Tests for `cellmaps_coembedding` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_constructor(self):
        """Tests constructor"""
        myobj = CellmapsCoEmbedder(outdir='foo')

        self.assertIsNotNone(myobj)

    def test_constructor_no_outdir(self):
        """ Tests run()"""
        try:
            myobj = CellmapsCoEmbedder()
            self.fail('Expected exception')
        except CellmapsCoEmbeddingError as ce:
            self.assertEqual('outdir is None', str(ce))

    def test_run_without_logging(self):
        """ Tests run() without logging."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = os.path.join(temp_dir, 'run')
            mock_embedding_generator = MagicMock()
            myobj = CellmapsCoEmbedder(outdir=run_dir,
                                       embedding_generator=mock_embedding_generator)
            try:
                myobj.run()
                self.fail('Expected CellmapsCoEmbeddingError')
            except CellmapsCoEmbeddingError as e:
                print(e)
                self.assertTrue('embeddings' in str(e))

            self.assertFalse(os.path.isfile(os.path.join(run_dir, 'output.log')))
            self.assertFalse(os.path.isfile(os.path.join(run_dir, 'error.log')))

        finally:
            shutil.rmtree(temp_dir)

    def test_run_with_logging(self):
        """ Tests run() with logging."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = os.path.join(temp_dir, 'run')
            mock_embedding_generator = MagicMock()
            myobj = CellmapsCoEmbedder(outdir=run_dir,
                                       embedding_generator=mock_embedding_generator,
                                       skip_logging=False)
            try:
                myobj.run()
                self.fail('Expected CellmapsCoEmbeddingError')
            except CellmapsCoEmbeddingError as e:
                print(e)
                self.assertTrue('embeddings' in str(e))

            self.assertTrue(os.path.isfile(os.path.join(run_dir, 'output.log')))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, 'error.log')))

        finally:
            shutil.rmtree(temp_dir)
