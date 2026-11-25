"""Repository ingestion: safe clone of one commit, scan, content-addressed file records."""


class PermanentIngestError(RuntimeError):
    """A failure that retrying cannot fix (too large, not a valid repo). The job fails at once."""
