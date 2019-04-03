from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='deepac',
      version='0.9',
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
      author_email='bartoszewiczj@rki.de',
      license='MIT',
      packages=['deepac', 'deepac.eval'],
      install_requires=[
          'keras>=2.2.4',
          'biopython',
          'scikit-learn',
          'matplotlib',
          'numpy',
          'h5py',
      ],
      entry_points={
          'console_scripts': ['deepac=deepac.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)