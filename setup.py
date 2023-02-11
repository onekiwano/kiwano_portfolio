from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

__version__ = '1.0.1'

setup(
    name='kiwano_portfolio',
    version=__version__,
    license='Apache License',
    url='https://github.com/onekiwano/kiwano_portfolio',
    author='Luca Herranz-Celotti',
    author_email='luca.herrtti@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='This is a python package to backtest, livetest and livetrade your strategies on crypto currencies.',
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)