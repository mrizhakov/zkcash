from pathlib import Path
from setuptools import find_packages, setup


requirements_file = Path(__file__).parent / "requirements.txt"

with requirements_file.open() as f:
    requirements = f.read().splitlines()


setup(
    name="tf_bionetta",
    include_package_data=True,
    package_data={
        "tf_bionetta": ["*.sh"]
    },
    packages=find_packages(include=["tf_bionetta", "tf_bionetta.*"]),
    install_requires=requirements,
    version="0.1.0",
    description="Implementation of TensorFlow layers and gadgets for Bionetta Protocol",
    author="Rarimo, Distributed Lab",
    license="MIT",
    url="https://github.com/rarimo/bionetta-tf",
)
