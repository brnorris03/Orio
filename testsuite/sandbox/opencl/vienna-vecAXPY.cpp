/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/*
* 
*   Tutorial: BLAS level 1 functionality (blas1.cpp and blas1.cu are identical, the latter being required for compilation using CUDA nvcc)
*
*/


// include necessary system headers
#include <iostream>

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

//include the generic inner product functions of ViennaCL
#include "viennacl/linalg/inner_prod.hpp"

//include the generic norm functions of ViennaCL
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"

// Some helper functions for this tutorial:
#include "Random.hpp"
#include "TAU.h"

int main(int argc, char * argv[])
{
  TAU_INIT(&argc, &argv)
  TAU_PROFILE_TIMER(orio_maintimer, "main()", "int (int, char **)", TAU_USER);
  TAU_PROFILE_START(orio_maintimer);
  TAU_PROFILE_SET_NODE(0);
    std::vector<viennacl::ocl::device> devices = viennacl::ocl::platform().devices();
    std::vector<cl_device_id> my_devices;
    my_devices.push_back(devices[0].id());
    viennacl::ocl::setup_context(0L, my_devices);

 
  //Change this type definition to double if your gpu supports that
  typedef double       ScalarType;
  
  /////////////////////////////////////////////////
  ///////////// Vector operations /////////////////
  /////////////////////////////////////////////////
  
  void * orio_profiler;
  TAU_PROFILER_CREATE(orio_profiler, "orio_generated_code", "", TAU_USER);

  for(int i = 0; i < 3; ++i) {

  //
  // Define a few vectors (from STL and plain C) and viennacl::vectors
  //
  std::vector<ScalarType>      std_vec1(1000000);
  std::vector<ScalarType>      std_vec2(1000000);

  viennacl::vector<ScalarType> vcl_vec1(1000000);
  viennacl::vector<ScalarType> vcl_vec2(1000000);
  viennacl::scalar<ScalarType> vcl_s1 = ScalarType(5.0);

  //
  // Let us fill the CPU vectors with random values:
  // (random<> is a helper function from Random.hpp)
  //
  
  for (unsigned int i = 0; i < 1000000; ++i)
  {
    std_vec1[i] = random<ScalarType>(); 
    std_vec2[i] = 0.0;
  }
  
  //
  // Copy the CPU vectors to the GPU vectors and vice versa
  //
    TAU_PROFILER_START(orio_profiler);   
  viennacl::copy(std_vec1.begin(), std_vec1.end(), vcl_vec1.begin()); //either the STL way
  viennacl::copy(std_vec2.begin(), std_vec2.end(), vcl_vec2.begin()); //either the STL way

  vcl_vec2 += vcl_s1 * vcl_vec1;

  viennacl::copy(vcl_vec2.begin(), vcl_vec2.end(), std_vec2.begin());

    TAU_PROFILER_STOP(orio_profiler);
  }
  double orio_inclusive[TAU_MAX_COUNTERS];
  TAU_PROFILER_GET_INCLUSIVE_VALUES(orio_profiler, &orio_inclusive);
  printf("{'/*@ coordinate @*/' : %g}\n", orio_inclusive[0]);
  return EXIT_SUCCESS;
}

