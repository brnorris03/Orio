# !/usr/bin/env python

# A Python script used for installing the Orio tool

# -----------------------------------------------------------

import glob
import setuptools

#with open("README.md", "r", encoding="utf-8") as fh:
with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(name='orio',
                 version='0.6.1',
                 description='ORIO -- An Annotation-Based Performance Tuning Tool',
                 author='Boyana Norris and Albert Hartono',
                 author_email='brnorris03@gmail.com',
                 maintainer='Boyana Norris',
                 maintainer_email='brnorris03@gmail.com',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='https://github.com/brnorris03/Orio',
                 packages=setuptools.find_packages(exclude=['test*']),
                 package_dir={'orio': 'orio'},
                 data_files=[('examples', glob.glob("examples/*"))],
                 scripts=['scripts/orcc', 'orf', 'orcuda', 'orcl'],
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires='>=3.6',
                 )
