"""Batch-processing specific exceptions."""


class SampleSkipError(Exception):
    """Raised for expected sample-level skips that are not pipeline failures."""

