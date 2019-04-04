import os
import pytest

skip_on_travis = pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="This test takes too long to run on Travis CI.")
