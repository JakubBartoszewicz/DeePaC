from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='deepac',
      version='0.12.2',
      description='Predicting pathogenic potentials of novel DNA with reverse-complement neural networks.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      keywords='deep learning DNA sequencing synthetic biology pathogenicity prediction',
      url='https://gitlab.com/rki_bioinformatics/DeePaC',
      author='Jakub Bartoszewicz',
      author_email='jakub.bartoszewicz@hpi.de',
      license='MIT',
      packages=['deepac', 'deepac.eval', 'deepac.tests', 'deepac.explain', 'deepac.gwpa'],
      python_requires='>=3',
      install_requires=[
          'tensorflow>=2.1',
          'biopython>=1.77',
          'scikit-learn>=0.22.1',
          'matplotlib>=3.1.3',
          'numpy>=1.17',
          'h5py>=2.10',
          'psutil>=5.6.7',
          'pandas>=1.0.3',
          'shap>=0.35',
          'weblogo>=3.7',
          'pybedtools>=0.8.1',
          'statsmodels>=0.11.0'
      ],
      entry_points={
          'console_scripts': ['deepac=deepac.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)
