"""
Pytest configuration for test suite.

This file allows us to configure test behavior globally.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def use_fast_model():
    """
    Use gpt-4o-mini for faster test execution.

    This fixture automatically runs before all tests and sets the default
    model to gpt-4o-mini which is much faster and cheaper than gpt-4.
    """
    # Temporarily override the default model for testing
    from obligationes import manager

    original_default = manager.DisputationConfig.__dataclass_fields__[
        "model_name"
    ].default
    manager.DisputationConfig.__dataclass_fields__["model_name"].default = "gpt-4o-mini"

    yield

    # Restore original default after tests
    manager.DisputationConfig.__dataclass_fields__["model_name"].default = (
        original_default
    )
