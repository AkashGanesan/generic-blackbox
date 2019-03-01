from setuptools import setup
from setuptools import find_packages

with open("README.org", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy',
                    'torch'
                    'setuptools']

tests_require = ['Pillow',
                 'requests']

# docs_require = ['sphinx >= 1.4',
#                 'sphinx_rtd_theme']

setup(name='BlackBoxDL',
      version='0.0.1',
      description='Custom DL models package ',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Akash Ganesan',
      author_email='akaberto@umich.edu',
      url='https://github.com/AkashGanesan/generic-blackbox',
      license='MIT',
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require={
          'tests': tests_require,
          'docs': docs_require
      },
      classifiers=[
            'Development Status :: Pre-Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
packages=find_packages())
