from setuptools import setup, find_packages

setup(
    name='gated_scnn',
    python_requires='>=3.7',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    # package_dir={"": "src"},
    install_requires=[
        'tensorflow-gpu==2.5.1',
        'imageio',
        'scipy',
    ],
    include_package_data=True,
    author='ben',
)
