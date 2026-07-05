from setuptools import setup, find_packages

setup(
    name="cv-vision-project",
    version="1.0.0",
    description="Object Detection with 5 models",
    author="Olesia Kosiakova",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "ultralytics>=8.0.0",
        "opencv-python",
        "matplotlib",
        "pandas",
        "numpy",
        "pyyaml",
        "tqdm",
    ],
)