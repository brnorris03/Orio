import sys
from os.path import abspath, dirname
import pytest

@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():
    # your setup code goes here, executed ahead of first test
    package_path = abspath(dirname(dirname(__file__)))
    sys.path.insert(0, package_path)
    print(sys.path)
