template_string='''
/*@ begin PerfTuning (       
  # 
  # @header_comment@
  #
  def build
  {
@build@
  }
   
  def performance_counter         
  {
@performance_counter@
  }
  
  def performance_params
  {
@performance_params@
@constraints@
  }

  def search
  {
@search@
  }
  
  def input_params
  {
@input_params@
  }
  
  def input_vars
  { 
@input_vars@
  }            
) @*/
'''

default_perf_params= dict(
    tiling=[1,8,16,32],
    unroll=[1]+list(range(2,17,2)),
    scalar_replacement=[False],
    vector=[False,True],
    openmp=[True],
)


default_params=dict(
    build=dict(build_command='gcc -g -O3',libs='-lm -lrt'),
    performance_counter=dict(repetitions=10),
    performance_params=dict(),
    constraints=dict(),
    search=dict(algorithm='Randomsearch', total_runs=1000),
    input_params=dict(),
    input_vars=dict(),
)


# Valid input types are static or dynamic, followed by double or float
default_input_type = 'dynamic double'
# default_input_type = 'static double'

# Not supported
# parallelize = ['False', 'True'],  # with standalone unrolljam
