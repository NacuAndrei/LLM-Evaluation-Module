from requirements import PACKAGES
from setuptools import setup, find_packages

setup(
    name="llm-eval-module",
    version="0.1",
    packages=find_packages(),
    install_requires=PACKAGES,
)
