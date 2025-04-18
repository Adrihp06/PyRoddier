from setuptools import setup, find_packages

setup(
    name="astrofisica-tfm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'PyQt5',
        'numpy',
        'astropy',
        'scipy',
        'matplotlib',
    ],
)