"""
DSGEpy — Dynamic Stochastic General Equilibrium Modeling Toolkit for Python.

A lightweight, pure-Python implementation of standard DSGE solution methods:
log-linearization, Blanchard-Kahn / Klein solvers, impulse response functions,
stochastic simulation, and Bayesian MCMC estimation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dsgepy",
    version="0.1.0",
    author="DSGEpy Contributors",
    description="Dynamic Stochastic General Equilibrium modeling toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/user/dsgepy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Science :: Economics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)
