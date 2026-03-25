from setuptools import setup, find_packages

setup(
    name="turboquant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy>=1.24", "scipy>=1.10"],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov>=4.0"],
        "bench": ["matplotlib"],
    },
    python_requires=">=3.10",
)
