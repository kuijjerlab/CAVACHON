from setuptools import setup

setup(
  name='CAVACHON',
  version='0.0.0',
  description="""Cell cluster Analysis with Variational Autoencoder using Conditional Hierarchy Of latent representioN""",
  url='https://github.com/dn070017/CAVACHONE',
  author='Ping-Han Hsieh',
  author_email='dn070017@gmail.com',
  license='MIT',
  packages=['cavachon'],
  install_requires=[
    'tensorflow==2.8.1',
    'tensorflow-probability==0.16.0',
    'scikit-learn>=1.1.2',
    'muon>=0.1.2',
    'gseapy==0.14.0',
    'mlflow>=1.26.1',
    'pandas>=1.4.1',
    'protobuf==3.20.0',
    'plotly>=5.8.2',
    'kaleido>=0.2.1',
    'pyyaml>=6.0',
  ],
  classifiers=[
    "Programming Language :: Python :: 3.8",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
  ],
  zip_safe=False,
  python_requires='>=3.8.0',
)
