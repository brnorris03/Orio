# Orio

Orio is an open-source extensible framework for the definition of domain-specific 
languages and generation of optimized code for multiple architecture targets, 
including support for empirical autotuning of the generated code.

For more detailed documentation, refer to the Orio website, https://brnorris03.github.io/Orio/.

## Installation

The Orio installation follows the standard Python setuptools process. The simplest way to install Orio is to run

```
pip install orio
```

For users who want to quickly install Orio to the standard locations of third-party Python modules (requiring superuser
privileges in a Unix system), simply append the `--user` option to the pip install line. Note that some modules may
require certain packages (e.g., Mlsearch requires `pandas`), so it's recommended that you
use [Conda](http://docs.conda.io) or a similar Python environment manager.

To test whether Orio has been properly installed in your system, try to execute `orcc` command as given below as an
example. If you used the
`--user` option, you can find `orcc` under your home directory, e.g., in `~/.local/bin` on Unix.

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

You can optionally include the installed orcc script location in the PATH shell variable. To do this for the above
example, the following two lines can be added in the .bashrc configuration file (assuming the user uses Bash shell, of
course).

```
export PYTHONPATH=$PYTHONPATH:/home/username/lib/python/site-packages
export PATH=$PATH:/home/username/bin
```

After making sure that the orcc executable is in your path, you can try some of the examples included in the testsuite
subdirectory, e.g.:

```
 $ cd examples
 $ orcc -v axpy5.c
```

The same directory contains two more examples of Orio input -- one with a separate tuning specification
file (`orcc -v -s axpy5.spec axpy5-nospec.c`) and another with two transformations specified using a Composite
annotation
(`orcc -v axpy5a.c`). To see a list of options, `orcc -h`. To keep all intermediate code versions, use the `-k` option.
You can also enable various levels of debugging output by using the `-d <NUM>` option, setting `<NUM>`
to an integer between 1 and
6, e.g., for the most verbose output `-d 6`. This is the recommended setting when submitting sample output for bug reports.

To use machine learning-based search (Mlsearch), install numpy, pandas, and scikit-learn modules. Alternatively, if
using conda, simply run `conda install pandas`
to obtain all prerequisites if needed.

If Orio reports problems building the code, adjust the compiler settings in the tuning spec included in the `axpy5.c`
example.

### Authors and Contact Information

Please report bugs at https://github.com/brnorris03/Orio/issues and include complete
examples that can be used to reproduce the errors. Send all other questions and comments to:
Boyana Norris, brnorris03@gmail.com .

Principal Authors:

* Boyana Norris, University of Oregon
* Albert Hartono, Intel
* Azamat Mametjanov, Argonne National Laboratory
* Prasanna Balaprakash, Argonne National Laboratory
* Nick Chaimov, University of Oregon
