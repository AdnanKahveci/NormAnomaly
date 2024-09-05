# setup.py

from setuptools import setup, find_packages

setup(
    name='AnomaFlow',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch==1.13.1',
        'torchvision==0.14.1',
        'numpy==1.24.0',
        'pandas==1.5.3',
        'scikit-learn==1.1.3',
        'matplotlib==3.6.2',
        'seaborn==0.12.1',
        'PyYAML==6.0',
        'tensorboard==2.11.0',
    ],
    entry_points={
        'console_scripts': [
            'anomaflow_train=src.train:main',
            'anomaflow_evaluate=src.evaluate:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='AnomaFlow: A framework for anomaly detection using normalizing flows.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AdnanKahveci/AnomaFlow',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
