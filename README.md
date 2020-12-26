Orio
====

Orio is an open-source extensible framework for the definition of domain-specific languages and generation of optimized code for multiple architecture targets, including support for empirical autotuning of the generated code.

For more detailed documentation, refer to the Orio website, http://brnorris03.github.io/Orio/.

Installation
========

The Orio installation follows the standard Python setuptools process. The simplest way 
to install Orio is to run 

```pip install orio
```

For users who want to quickly install Orio to the standard locations
of third-party Python modules (requiring superuser privileges in a
Unix system), the installation is straightforward as shown below. Note that 
some modules may require certain packages (e.g., Mlsearch requires `pandas`), 
so it's recommended that you use [Conda](http://docs.conda.io) or a similar Python 
environment manager. 

```
  $ tar -xzf orio.tar.gz
  $ cd orio
  $ python setup.py install --prefix=$VALID_PYTHON_PATH
```

On a Unix platform, the install command without the `--prefix` option will install Orio in the 
default python installation (system version or currently activated virtual environment). An alternetive
approach is to install using the `--user` option, which does not modify the python installation. 
Without any options, `python setup.py install` will typically require superuser permissions and 
would install the Orio executables (orcc, orcu, orcl, etc.) in `/usr/local/bin` and 
and python packages in `/usr/local/lib/python2.X`. At this point, there is no uninstall script,
so removing the above two components manually is sufficient to uninstall Orio.

To test whether Orio has been properly installed in your system, try
to execute `orcc` command as given below as an example.

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
separate tuning specification file (`orcc -v -s axpy5.spec axpy5-nospec.c`) and
another with two transformations specified using a Composite annotation
(`orcc -v axpy5a.c`). To see a list of options, `orcc -h`. To keep all intermediate code
versions, use the `-k` option. You can also enable various levels of debugging 
output by setting the ORIO_DEBUG_LEVEL to an integer value betwen 1 and 6, e.g., for 
the most verbose output `export ORIO_DEBUG_LEVEL=6` and run Orio with the `-v` 
command-line option. This is the recommended setting when submitting sample output for
bug reports.


To use machine learning-based search (Mlsearch), install numpy, pandas, 
and scikit-learn modules. Alternatively, if using conda, simply run `conda install pandas`
to obtain all prerequisites if needed.


If Orio reports problems building the code, adjust the compiler settings in 
the tuning spec included in the `axpy5.c` example.

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
