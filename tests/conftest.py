import sys
import pytest
from firefly_uas.location.model import LocOptModel


# each test runs on cwd to its temp dir
@pytest.fixture(autouse=True)
def go_to_tmpdir(request):
    # Get the fixture dynamically by its name.
    tmpdir = request.getfixturevalue("tmpdir")
    # ensure local test created packages can be imported
    sys.path.insert(0, str(tmpdir))
    # Chdir only for the duration of the test.
    with tmpdir.as_cwd():
        yield

def test_loc_opt_model_initialization():
    """
    Test location model initialization
    """
    model = LocOptModel()
    assert model is not None
