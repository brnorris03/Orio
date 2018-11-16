Orio
====

Orio is an open-source extensible framework for the definition of domain-specific languages and generation of optimized code for multiple architecture targets, including support for empirical autotuning of the generated code.

For more detailed documentation, refer to the Orio website, http://brnorris03.github.io/Orio/.

Installation
========

The Orio installation follows the standard Python Module Distribution
Utilities, or Disutils for short.

For users who want to quickly install Orio to the standard locations
of third-party Python modules (requiring superuser privileges in a
Unix system), the installation is straightforward as shown below.

```
  $ tar -xzf orio.tar.gz
  $ cd orio
  $ python setup.py install
```

On a Unix platform, the above install command will normally put an
orcc script in the /usr/bin location, and also create an orio module
directory in the /usr/lib/python2.X/site-packages location. You can install
Orio in a different location by specifying the --prefix option to the setup.py 
script.

To test whether Orio has been properly installed in your system, try
to execute orcc command as given below as an example.

```
  $ orcc --help

  description: compile shell for Orio

  usage: orcc [options] <ifile>
    <ifile>   input file containing the annotated code

  options:
    -h, --help                     display this message
    -o <file>, --output=<file>     place the output to <file>
    -v, --verbose                  verbosely show details of the results of the running program
```

In order to install Orio to an alternate location, users need to
supply a base directory for the installation. For instance, the
following command will install an orcc script under
/home/username/bin, and also put an orio module under
/home/username/lib/python/site-packages.

```
  $ tar -xvzf orio.tar.gz
  $ cd orio
  $ python setup.py install --prefix=/home/username
```

You can optionally include the installed orcc script location in the PATH
shell variable. 
To do this for the above example, the following two
lines can be added in the .bashrc configuration file (assuming the
user uses Bash shell, of course).

```
export PYTHONPATH=$PYTHONPATH:/home/username/lib/python/site-packages
export PATH=$PATH:/home/username/bin
```

After making sure that the orcc executable is in your path, you can 
try some of the examples included in the testsuite subdirectory, e.g.:

```
 $ cd examples
 $ orcc -v axpy5.c
```

The same directory contains two more examples of Orio input -- one with a 
separate tuning specification file (orcc -v -s axpy5.spec axpy5-nospec.c) and
another with two transformations specified using a Composite annotation
(orcc -v axpy5a.c).


To use machine learning based search (Mlsearch), install numpy, panda, scikit-learn modules


If Orio reports problems building the code, adjust the compiler settings in 
the tuning spec included in the axpy5.c.

Authors and Contact Information
=========================

  Please send all questions, bugs reports, and comments to:
    Boyana Norris, brnorris03@gmail.com
    
 Principal Authors:
 
 * Boyana Norris, University of Oregon
 * Albert Hartono, Intel 
 * Azamat Mametjanov, Argonne National Laboratory
 * Prasanna Balaprakash, Argonne National Laboratory
 * Nick Chaimov, University of Oregon
