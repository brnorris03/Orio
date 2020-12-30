# Orio

Orio is an open-source extensible framework for the definition of domain-specific 
languages and generation of optimized code for multiple architecture targets, 
including support for empirical autotuning of the generated code.

For more detailed documentation, refer to the Orio website, https://brnorris03.github.io/Orio/.

## Installation

Orio is implemented in Python 3. Some search methods (e.g., Mlsearch) require the pandas and sklearn 
packages.  The simplest way to install Orio is to run

```
pip install orio
```

This will install the most recent release of Orio and the packages it
uses in your current Python environment. You can also
add the `--user` option if the above command requires superuser privileges.

If you want to build Orio from a git clone, you can use `pip install -e .` in the top-level directory. 
Note that you can simply run `orcc` (and the other top-level command-line scripts) directly from the git clone 
without installing anything. Testing is provided through pydev, to run all available
tests, run`pytest` or `pytest -v` in the top-level Orio directory. 

To test whether Orio has been properly installed in your system, try to execute `orcc` 
command as given below as an example. If you used the
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

After making sure that the `orcc` executable is in your path, you can try some of the examples included in the testsuite
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

### Publications

* B. Norris, A. Hartono, and W. Gropp. Annotations for productivity and performance portability. Petascale Computing: Algorithms and Applications, pp. 443–462. Chapman & Hall / CRC Press, Taylor and
Francis Group, Computational Science, 2007, http://www.mcs.anl.gov/uploads/cels/
papers/P1392.pdf. 

* Azamat Mametjanov, Daniel Lowell, Ching-Chen Ma, and Boyana Norris. 2012. Autotuning Stencil-Based Computations on GPUs. In Proceedings of the 2012 IEEE International Conference on Cluster Computing (CLUSTER '12). IEEE Computer Society, USA, 266–274. DOI:https://doi.org/10.1109/CLUSTER.2012.46

* Prasanna Balaprakash, Stefan M. Wild, Boyana Norris,
SPAPT: Search Problems in Automatic Performance Tuning,
Procedia Computer Science,
Volume 9, 2012, Pages 1959-1968, ISSN 1877-0509,
https://doi.org/10.1016/j.procs.2012.04.214.

* N. Chaimov, B. Norris, and A. Malony. Toward multi-target autotuning for accelerators. Proceedings of
the 20th IEEE International Conference on Parallel and Distributed Systems, December 16-19, 2014,
Hsinchu, Taiwan, 2014, http://ix.cs.uoregon.edu/~norris/icpads14.pdf.

* Lim, Robert V., B. Norris and A. Malony. “Autotuning GPU Kernels via Static and Predictive Analysis.” 2017 46th International Conference on Parallel Processing (ICPP) (2017): 523-532. https://arxiv.org/pdf/1701.08547


### Old websites

* Orio's old webpage:
  http://trac.mcs.anl.gov/projects/performance/wiki/Orio

* Some hidden links:
  http://trac.mcs.anl.gov/projects/performance/wiki/AnnPerformance  (Old results for tuning Pluto)

