import setuptools

setuptools.setup(
    name="pib_lib",
    version="0.0.1",
    author="Valentin Wyss",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"":"."},
    packages=setuptools.find_packages(where="pib_lib"),
    python_requires=">=3.6",
    install_requires = ["numpy", "scipy", "matplotlib"]
)