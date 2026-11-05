#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_coembedding` package."""

import os
import tempfile
import shutil

import unittest

from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError

from cellmaps_coembedding import cellmaps_coembeddingcmd


class TestCellmapsCoEmbedding(unittest.TestCase):
    """Tests for `cellmaps_coembedding` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_parse_arguments(self):
        """Tests parse arguments"""
        res = cellmaps_coembeddingcmd._parse_arguments('hi', ['foo',
                                                               '--ppi_embeddingdir',
                                                               'apms',
                                                               '--image_embeddingdir',
                                                               'image'])

        self.assertEqual(res.verbose, 1)
        self.assertEqual(res.logconf, None)

        someargs = ['-vv', '--logconf', 'hi',
                    'foo',
                    '--ppi_embeddingdir',
                    'apms',
                    '--image_embeddingdir',
                    'image']
        res = cellmaps_coembeddingcmd._parse_arguments('hi', someargs)

        self.assertEqual(3, res.verbose)
        self.assertEqual('hi', res.logconf)
        self.assertEqual('foo', res.outdir)
        self.assertEqual('apms', res.ppi_embeddingdir)
        self.assertEqual('image', res.image_embeddingdir)

    def test_main(self):
        """Tests main function"""

        # try where loading config is successful
        try:
            temp_dir = tempfile.mkdtemp()
            res = cellmaps_coembeddingcmd.main(['myprog.py',
                                                'foo',
                                                '--ppi_embeddingdir',
                                                'apms',
                                                '--image_embeddingdir',
                                                'image'])
            self.assertEqual(2, res)
        finally:
            shutil.rmtree(temp_dir)

    def test_main_with_embedding_dirs_two_dirs(self):
        try:
            temp_dir = tempfile.mkdtemp()
            res = cellmaps_coembeddingcmd.main(['myprog.py',
                                                'foo',
                                                '--embeddings',
                                                'dir1', 'dir2'])
            self.assertIn(res, [0, 2])
        finally:
            shutil.rmtree(temp_dir)

    def test_main_with_embedding_dirs_more_than_two_dirs(self):
        with self.assertRaises(CellmapsCoEmbeddingError):
            cellmaps_coembeddingcmd.main(['myprog.py',
                                          'foo',
                                          '--embeddings',
                                          'dir1', 'dir2', 'dir3'])

    def test_main_with_conflicting_flags(self):
        with self.assertRaises(CellmapsCoEmbeddingError):
            cellmaps_coembeddingcmd.main(['myprog.py',
                                          'foo',
                                          '--ppi_embeddingdir', 'apms',
                                          '--image_embeddingdir', 'image',
                                          '--embeddings', 'dir1', 'dir2'])

    def test_main_with_old_flags_only(self):
        try:
            temp_dir = tempfile.mkdtemp()
            res = cellmaps_coembeddingcmd.main(['myprog.py',
                                                'foo',
                                                '--ppi_embeddingdir', 'apms',
                                                '--image_embeddingdir', 'image'])
            self.assertIn(res, [0, 2])
        finally:
            shutil.rmtree(temp_dir)

    def test_main_with_no_flags(self):
        with self.assertRaises(CellmapsCoEmbeddingError):
            cellmaps_coembeddingcmd.main(['myprog.py',
                                          'foo'])
