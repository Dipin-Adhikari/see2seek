from setuptools import setup, find_packages

setup(
    name="see_to_seek",
    version="0.1.0",
    description="DINOv2 observation encoder for zero-shot object navigation (RoboTHOR)",
    author="Dipin",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "open_clip_torch>=2.20.0",
        "ai2thor>=4.3.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "pyyaml>=6.0",
    ],
)