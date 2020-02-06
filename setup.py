from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='deepac',
      version='0.9.3',
      description='Predicting pathogenic potentials of novel DNA with reverse-complement neural networks.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      keywords='deep learning DNA sequencing synthetic biology pathogenicity prediction',
      url='https://gitlab.com/rki_bioinformatics/DeePaC',
      author='Jakub Bartoszewicz',
      author_email='jakub.bartoszewicz@hpi.de',
      license='MIT',
      packages=['deepac', 'deepac.eval', 'deepac.tests', 'deepac.explain', 'deepac.gwap'],
      python_requires='>=3',
      install_requires=[
          'keras>=2.2.4',
          'tensorflow',
          'biopython>=1.76',
          'scikit-learn',
          'matplotlib',
          'numpy',
          'h5py',
          'psutil',
          'pandas',
          'shap>=0.34',
          'weblogo>=3.7.4',
          'pybedtools',
          'statsmodels'
      ],
      entry_points={
          'console_scripts': ['deepac=deepac.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)
