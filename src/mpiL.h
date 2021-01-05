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
#define AT_MPI_BUFMAX 25000000
#define AT_MPI_ROOT_RANK 0

void derive_MPI_OPTIONS( void );
void derive_MPI_EXAMPLE( void );
void derive_MPI_TREENODE( void );
void broadcast_options(Args_Opts *Args);
void _broadcast_subset(CV_Subset *sub, int myrank, Args_Opts args);
void receive_subset(CV_Subset *sub, int myrank, Args_Opts args);
void send_subset(CV_Subset *sub, int send_to, Args_Opts args);
void receive_one_tree(char *buf, int *pos, DT_Node **tree, int num_classes, int nodes);
void send_trees(DT_Node **trees, Tree_Bookkeeping *books, int num_classes, int my_num);
int check_mpi_error(int err, char *label);

typedef enum {
    MPI_SUBSETMETA_TAG = 0,
    MPI_SUBSETDATA_TAG = 1,
    MPI_PERM2SEND_TAG = 2,
    MPI_TREENODE_TAG = 3,
    MPI_VOTE_CACHE_TAG = 4
} MPI_TAG_TYPE;

typedef enum {
    AT_SEND_NO_TREE,
    AT_SEND_ONE_TREE,
    AT_RECEIVE_NEW_DATA
} AT_TREE_PASSING_CODE;
