from setuptools import setup, find_packages

setup(
    author="Siem de Jong",
    name="dpat",
    packages=find_packages(include=["dpat", "dpat.*"]),
    entry_points={
        "console_scripts": [
            "dpat=dpat.cli:main",
        ],
    },
)