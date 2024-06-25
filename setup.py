from setuptools import setup
from setuptools import find_packages

setup(
    name='rec_adam',
    version='0.0',
    install_requires=[
        'torch',
        'transformers',
        'numpy',
    ],
    packages=find_packages(),
    # packages=['ai_competition', 'machine_learning'],
    # package_dir={
    #     'ai_competition': 'ai_competition',
    #     'machine_learning': 'ai_competition',
    # },
    zip_safe=False,
)
