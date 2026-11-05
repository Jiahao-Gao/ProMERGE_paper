#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration Tests for `cellmaps_coembedding` package."""

import os

import unittest
from cellmaps_coembedding import cellmaps_coembeddingcmd

SKIP_REASON = 'CELLMAPS_COEMBEDDING_INTEGRATION_TEST ' \
              'environment variable not set, cannot run integration ' \
              'tests'

@unittest.skipUnless(os.getenv('CELLMAPS_COEMBEDDING_INTEGRATION_TEST') is not None, SKIP_REASON)
class TestIntegrationCellmaps_generate_ppi(unittest.TestCase):
    """Tests for `cellmaps_coembedding` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_something(self):
        """Tests parse arguments"""
        self.assertEqual(1, 1)
