from setuptools import setup, find_packages

setup(
    name="groomrl",
    version="2.0.0",
    description="GroomRL with SB3 + XRL support",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn",
        "gymnasium",
    ],
    extras_require={
        "sb3":    ["stable-baselines3>=2.0"],
        "xrl":    ["torch", "torch_geometric", "networkx"],
        "legacy": ["keras", "tensorflow", "keras-rl"],
    },
    entry_points={
        "console_scripts": [
            "groomer=groomrl.scripts.groomer:main",
        ]
    },
)
