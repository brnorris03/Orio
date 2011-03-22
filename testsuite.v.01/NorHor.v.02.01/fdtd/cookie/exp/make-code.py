#!/usr/bin/env python

import sys, os

if len(sys.argv) < 3:
    print 'error: missing arguments (need 2 arguments)'
    sys.exit(1)

f=open('template.begin.c') 
begin_code=f.read()
f.close()

f=open('template.end.c') 
end_code=f.read()
f.close()

f=open(sys.argv[1])
body_code=f.read()
f.close()

code=begin_code+body_code+end_code
f=open(sys.argv[2],'w')
f.write(code)
f.close()

