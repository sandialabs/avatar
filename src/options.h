/********************************************************************************** 
Avatar Tools 
Copyright (c) 2019, National Technology and Engineering Solutions of Sandia, LLC
All rights reserved. 

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer  in the
  documentation and/or other materials provided with the distribution.


3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For questions, comments or contributions contact 
Philip Kegelmeyer, wpk@sandia.gov 
*******************************************************************************/
#ifndef __OPTIONS__
#define __OPTIONS__

Args_Opts process_opts(int argc, char **argv);
void late_process_opts(int num_atts, int num_examples, Args_Opts *args);
void set_output_filenames(Args_Opts *args, Boolean force_input, Boolean force_output);
void free_Args_Opts(Args_Opts args);
void free_Args_Opts_Full(Args_Opts args);
void read_classes_file(CV_Class *class, Args_Opts *args);
int sanity_check(Args_Opts *args);
void read_partition_file(CV_Partition *partition, Args_Opts *args);
void print_skewed_per_class_stats(FILE *fh, char *comment, Args_Opts args, CV_Metadata meta);
void display_dtmpi_opts(FILE *fh, char *comment, Args_Opts args);
void display_dt_opts(FILE *fh, char *comment, Args_Opts args, CV_Metadata meta);
void display_cv_opts(FILE *fh, char *comment, Args_Opts args);
void display_fv_opts(FILE *fh, char *comment, Args_Opts args);

#endif // __OPTIONS__
