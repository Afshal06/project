from setuptools import setup, find_packages

setup(
    name='leaf-regeneration-classification',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for the regeneration and classification of leaves using machine learning.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',  # or 'torch' if using PyTorch
        'opencv-python',
        'matplotlib',
        'seaborn',
        'flask',  # for API serving
        'pyyaml',
        'jupyter',  # for notebooks
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)