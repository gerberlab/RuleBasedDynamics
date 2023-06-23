from setuptools import setup 

setup(name='dynrules',
    version='0.1.0',
    description='Dynamical rule model',
    author='Gary Uppal',
    packages=['dynrules'],
    install_requires=[
                    'numpy',
                    'scikit-learn',
                    'matplotlib',
                    'seaborn',
                    'pandas',
                    'scipy',
                    'jupyterlab',                    
                    ],
    python_requires=">=3.7",
)