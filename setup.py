from setuptools import setup, find_packages
from pathlib import Path

# Read dependencies from requirements.txt
root = Path(__file__).parent
with open(root / "requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pmcmc",
    version="0.1.0",
    author="James Green",
    description="Particle MCMC and state-space model implementations",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires="==3.12.3",
    install_requires=requirements,
)

