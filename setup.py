#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

#import Declic Experiment Analysing Tool


setup(
   name='DEAT',
   version= "0.0.1",
   
   packages=find_packages(),
   
   author='Samuel Quiret',
   author_email='samuel.quiret@univ-amu.fr',
   description='This package uses Jorge Pereda\'s work to offer tools to perform\
   a full analysis on the images obtained with the Declic experiment',
   
   
   install_requires=['opencv-python', 'scikit-image', 'scikit-learn==0.18.1',
                      'Pillow',  'matplotlib==2.0.0', 'scipy==1.1.0', 'numpy >= 1.13'],
   include_package_data=True,
   #url='http://github.com/sametmax/sm_lib',
   
   classifiers=[
        "Programming Language :: Python",
        "Development Status :: 1 - Development",
        "Natural Language :: French",
        "Operating System :: Windows",
        "Programming Language :: Python :: 2.7",
        "Topic :: Research",
    ],
   
           
   license="IM2NP",
   long_description=open('README.md').read(),
   
)

print('\nInstallation success!!\n')