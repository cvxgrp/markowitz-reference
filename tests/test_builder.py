# """
# testing the builder
# """
import os.path

from markowitz import foo


def test_foo(resource_dir):
    assert os.path.exists(resource_dir)
    assert foo() == "bar"
