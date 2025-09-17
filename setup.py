from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jepa-pytorch",
    version="0.1.0",
    author="Benjamin Seguin",
    author_email="benjamin.seguin@outlook.fr",
    description="PyTorch implementation of Joint-Embedding Predictive Architecture (JEPA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BenjaminSeguin/jepa-pytorch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="jepa, self-supervised-learning, pytorch, representation-learning, deep-learning",
)