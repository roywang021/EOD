#!/usr/bin/env python
from setuptools import setup

setup(
    name="EOD",
    version=0.1,
    author="roywang021",
    url="https://github.com/roywang021/EOD",
    description="Codebase for Evidential Object Detection",
    python_requires=">=3.6",
    install_requires=[
        'timm', 'opencv-python'
    ],
)
