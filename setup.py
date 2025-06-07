import platform

import setuptools
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()
    descr_lines = long_description.split("\n")
    descr_no_gifs = []  # gifs are not supported on PyPI web page
    for dl in descr_lines:
        if not ("<img src=" in dl and "gif" in dl):
            descr_no_gifs.append(dl)

    long_description = "\n".join(descr_no_gifs)


def is_macos():
    return platform.system() == "Darwin"


setup(
    # Information
    name="NEAT",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.1",
    url="https://github.com/BartekCupial/NEAT",
    author="Bartłomiej Cupiał",
    license="MIT",
    keywords="reinforcement learning neural evolution jax",
    project_urls={
        "Github": "https://github.com/BartekCupial/NEAT",
    },
    install_requires=[
        "opencv-python",
        "networkx",
        "hydra-core",
    ],
    extras_require={
        # some tests require Atari and Mujoco so let's make sure dev environment has that
        "dev": ["black", "isort>=5.12", "pytest<8.0", "flake8", "pre-commit", "twine"]
    },
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./", include=["neat*"]),
    include_package_data=True,
    python_requires=">=3.8",
)
