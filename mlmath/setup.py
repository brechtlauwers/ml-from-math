from setuptools import setup, find_packages

setup(
    name="mlmath",
    version="0.1",
    author="Brecht Lauwers",
    description="Building machine learning models from pure math",
    url="https://github.com/brechtlauwers/ml-from-math",
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
)
