#!/usr/bin/env python

import os, sys

f = os.popen('ls')
output = f.read()
f.close()
files=output.split('\n')
files=[f.strip() for f in files]
for f in files:
    if f.endswith('.eps'):
        os.system('epstopdf %s' % f)

