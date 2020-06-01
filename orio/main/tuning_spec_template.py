template_string='''
/*@ begin PerfTuning (       
  # 
  # @header_comment@
  #
  def build
  {
    arg build_command = @build_command@;
    arg libs = @libs@;
  }
   
  def performance_counter         
  {
    arg repetitions = @reps@;
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