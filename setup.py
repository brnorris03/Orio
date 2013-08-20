#!/usr/bin/env python

# A Python script used for installing the Orio tool

#-----------------------------------------------------------

import os
from distutils.core import setup

#-----------------------------------------------------------

# to traverse the source code directory to get all python packages
py_packages = []
cur_dir = os.getcwd()
src_dir = os.path.join(cur_dir, 'orio')
for root, dirs, files in os.walk(src_dir, topdown=True):
    if '__init__.py' in files:
        rel_dir = root[len(cur_dir)+1:]
        dir_names = rel_dir.split(os.sep)
        py_packages.append('.'.join(['orio'] + dir_names[1:]))

print py_packages

#-----------------------------------------------------------

# to remove certain packages not included in the source distribution
if False:
    removed_packages = ['orio.module.polysyn', 'orio.module.spmv', 
                        'orio.module.align', 'orio.module.loop', 
                        'orio.module.simplyrewrite', 'orio.module.pragma',
                        'orio.module.pluto', ]
    n_py_packages = []
    for p in py_packages:
        is_removed = False
        for r in removed_packages:
            if p.startswith(r):
                is_removed = True
        if not is_removed:
            n_py_packages.append(p)
    py_packages = n_py_packages

#-----------------------------------------------------------

# make a call to the setup function
setup(name = 'orio',
      version = '0.2.2',
      description = 'ORIO -- An Annotation-Based Performance Tuning Tool',
      author = 'Boyana Norris and Albert Hartono',
      author_email = 'norris@mcs.anl.gov',
      maintainer = 'Boyana Norris',
      maintainer_email = 'norris@mcs.anl.gov',
      url = 'https://trac.mcs.anl.gov/projects/performance/wiki/Orio',
      packages = py_packages,
      package_dir = {'orio' : 'orio'},
      #package_data = {'orio' : ['tool/zestyparser/*']},
      scripts = ['orcc', 'orf', 'orcuda', 'orcl'])


