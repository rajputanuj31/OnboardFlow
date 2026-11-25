"""Shared constants for unit tests."""

# Port 1 on loopback refuses connections immediately; the password is a canary that
# must never show up in logs, reprs or API responses.
TEST_DB_URL = "postgresql+asyncpg://onboardflow:s3cret-pw@127.0.0.1:1/onboardflow"
