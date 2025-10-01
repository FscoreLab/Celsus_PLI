from setuptools import find_packages, setup

setup(
    name="fvlm",
    packages=find_packages(),
    version="0.1.0",
    description="fVLM",
    license="",
    install_requires=[
        'TotalSegmentator==2.11.0',
        'monai>=1.3.2',
        'timm>=0.4.12',
        'opencv-python>=4.10.0',
        'omegaconf>=2.3.0',
        'iopath>=0.1.10',
        'decord>=0.6.0',
        'webdataset>=0.2.100',
    ]
)
