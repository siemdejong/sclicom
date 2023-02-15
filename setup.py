from setuptools import find_packages, setup  # type: ignore

__version__ = "1.4.0"

# Get the long description from the README file
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    author="Siem de Jong",
    author_email="siem.dejong@hotmail.nl",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        "console_scripts": [
            "dpat=dpat.cli:cli",
        ],
    },
    install_requires=[
        "dlup>=0.3",
        "scikit-learn>=0.24",
    ],
    license="GNU General Public License v3",
    keywords="dpat",
    name="dpat",
    packages=find_packages(include=["dpat", "dpat.*"]),
    url="https://github.com/siemdejong/hhg-dpat",
    version=__version__,
)
