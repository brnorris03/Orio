# Introduction

This is a simple Orio autotuning example, which includes a custom performance testing skeleton that is instrumented with [Caliper](https://github.com/LLNL/Caliper) for flexible collection of detailed performance data. Note that the autotuner is only relying on simple timing, which in this case includes the Caliper overhead. 

# Prerequisites

The following must be available/installed on your machine to test this example:

* [Orio](https://github.com/brnorris03/Orio)

* [Caliper](https://github.com/LLNL/Caliper)
* gcc

# Running the example

First, edit the `build` section of tuning spec `mm_tune.c` and make sure the build command and paths to the Caliper include and library paths are correct for your installation:

```
   def build {
      arg build_command = 'gcc-9 -I$HOME/soft/caliper/include -g -fopenmp -mcmodel=large @CFLAGS';
      arg libs = '-L$HOME/soft/caliper/lib -Wl,-rpath,$HOME/soft/caliper/lib -lcaliper';
    } 
```

Optionally change any other settings, e.g., problem sizes in the `input_params` section. 

To run, use the full-path command for the `orcc` script, or add its top-level source or `installation/bin` directory to your path:

​	```orcc mm_tune.c```

For more verbose output, use the `-v` option. In addition, for debugging, you may want to keep intermediate generated files (names starting with `__orio`) with the `-k` option. In addition, you can indicate that Orio should exit for all errors (by default, some errors are ignored and the code version that caused them is not considered):

​	```orcc -vk --stop-on-error mm_tune.c```

Upon finishing, Orio will report the best parameters for each code size, and create a file `_mm_tune.c` containing the best-performing code versions for each problem size.

