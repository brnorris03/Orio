# Introduction

This is a simple Orio autotuning example, which includes a custom performance testing skeleton that is instrumented with [Caliper](https://github.com/LLNL/Caliper) for flexible collection of detailed performance data. Note that the autotuner is only relying on simple timing, which in this case includes the Caliper overhead. 

# Prerequisites

The following must be available/installed on your machine to test this example:

* [Orio](https://github.com/brnorris03/Orio) You can simply clone Orio and run the `Orio/orcc` command directly, or you can optionally install it with `python2 setup.py install --prefix=INSTALLDIR`.
* [Caliper](https://github.com/LLNL/Caliper). The configuration here relies on having [PAPI](http://icl.utk.edu/papi/software/), as well and configuring Caliper with it, e.g., `cmake -DCMAKE_INTALL_PREFIX=$CALIPER_DIR -DWITH_PAPI=/usr/local ..`.
* gcc

# Running the example

First, edit the `build` section of tuning spec `mm_tune.c` and make sure the build command and paths to the Caliper include and library paths are correct for your installation. Alternatively, you can simply define the `CALIPER_DIR` environment variable to point to the top-level Caliper installation directory.

```
   def build {
      arg build_command = 'gcc-9 -I$CALIPER_DIR/include -g -fopenmp -mcmodel=large @CFLAGS';
      arg libs = '-L$CALIPER_DIR/lib -Wl,-rpath,$CALIPER_DIR/lib -lcaliper';
    } 
```

Optionally change any other settings, e.g., problem sizes in the `input_params` section. 

To run, use the full-path command for the `orcc` script, or add its top-level source or `installation/bin` directory to your path:

​	```orcc mm_tune.c```

For more verbose output, use the `-v` option. In addition, for debugging, you may want to keep intermediate generated files (names starting with `__orio`) with the `-k` option. In addition, you can indicate that Orio should exit for all errors (by default, some errors are ignored and the code version that caused them is not considered):

​	```orcc -vk --stop-on-error mm_tune.c```

Upon finishing, Orio will report the best parameters for each code size, and create a file `_mm_tune.c` containing the best-performing code versions for each problem size.

It will also create a file called `caliper.log`, containing all the measurements, with separate sections for each code version. For example:

```
__orio_perftest1.exe: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[
{"event.begin#function":"main","papi.PAPI_DP_OPS":0},
{"event.begin#loop":"reps_loop","path":"main","papi.PAPI_DP_OPS":0},
{"path":"main/reps_loop","event.begin#iteration#reps_loop":0,"papi.PAPI_DP_OPS":2},
{"path":"main/reps_loop","event.end#iteration#reps_loop":0,"iteration#reps_loop":0,"papi.PAPI_DP_OPS":268435456},
{"event.begin#annotation":"validation","path":"main/reps_loop","papi.PAPI_DP_OPS":3},
{"event.end#annotation":"validation","path":"main/reps_loop/validation","papi.PAPI_DP_OPS":0},
{"path":"main/reps_loop","event.begin#iteration#reps_loop":1,"papi.PAPI_DP_OPS":2},
{"path":"main/reps_loop","event.end#iteration#reps_loop":1,"iteration#reps_loop":1,"papi.PAPI_DP_OPS":268435456},
{"path":"main/reps_loop","event.begin#iteration#reps_loop":2,"papi.PAPI_DP_OPS":5},
{"path":"main/reps_loop","event.end#iteration#reps_loop":2,"iteration#reps_loop":2,"papi.PAPI_DP_OPS":268435456},
{"path":"main/reps_loop","event.begin#iteration#reps_loop":3,"papi.PAPI_DP_OPS":5},
{"path":"main/reps_loop","event.end#iteration#reps_loop":3,"iteration#reps_loop":3,"papi.PAPI_DP_OPS":268435456},
{"path":"main/reps_loop","event.begin#iteration#reps_loop":4,"papi.PAPI_DP_OPS":5},
{"path":"main/reps_loop","event.end#iteration#reps_loop":4,"iteration#reps_loop":4,"papi.PAPI_DP_OPS":268435456},
{"event.end#loop":"reps_loop","path":"main/reps_loop","papi.PAPI_DP_OPS":3},
{"event.end#function":"main","path":"main","papi.PAPI_DP_OPS":0}
]
```

The integer array represents the coordinate corresponding to the parameter values tested in this example; the `tuning*.log` generated during the same run contains the mapping between these integer indices and actual values. You can quickly look them up with:

```
$ grep "0, 0, 0, 0, 0, 0, 0, 0, 0, 0" tuning_mm_tune.c_1876850.log  | grep perf_params
(run 1) | {"compile_time": 0.0, "run": 1, 
	"cost": [21.322, 22.8436, 18.7114, 19.4075, 18.0779], 
	"coordinate": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	"perf_params": {"T1_Ja": 1, "SCREP": false, "T1_J": 1, "T1_K": 1, "U_K": 1, 
	"U_J": 1, "T1_Ka": 1, "OMP": true, "VEC": false, "CFLAGS": "-O3"}, 
	"transform_time": 0.0}
```

where `tuning_mm_tune.c_1876850.log` is your current/most recent log produced by orcc.

# Miscellaneous

Use the `./cleanup.sh` script to remove all generated files before rerunning.