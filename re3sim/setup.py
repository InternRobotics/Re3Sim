from setuptools import setup, find_packages

setup(
    name="real2sim2real",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        "trimesh",
        "mplib",
        "pathos",
        "tqdm",
        "h5py",
        "einops",
        "wandb",
        "ipython",
        "packaging",
        "opencv-python",
        "flask",
        "urdfpy",
        "pin",
        "open3d",
        "roboticstoolbox-python",
        "pyquaternion",
        "scipy==1.14.1",
    ],
)
