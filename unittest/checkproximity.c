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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/rw_data.h"
#include "../src/tree.h"
#include "../src/options.h"
#include "../tools/proximity_utils.h"

static CV_Subset Subset;
static DT_Ensemble Ensemble;
static Args_Opts Args;

void _read_data_and_trees( void );
void _read_data_and_trees() {
    AV_SortedBlobArray Sorted_Examples;
    FC_Dataset ds = {0};
    CV_Dataset Dataset = {0};

    memset(&Args, 0, sizeof(Args_Opts));
    memset(&Subset, 0, sizeof(CV_Subset));
    memset(&Ensemble, 0, sizeof(DT_Ensemble));
    Args.caller = PROXIMITY_CALLER;
    Args.format = AVATAR_FORMAT;
    Args.base_filestem = strdup("proximity_test");

    Args.data_path = strdup("./data");
    Args.datafile = strdup("./data/proximity_test.data");
    Args.truth_column = 5;
    Args.save_trees = TRUE; // This prevents the dotted temp filename from being used for the trees file

    set_output_filenames(&Args, TRUE, TRUE);
    av_exitIfError(av_initSortedBlobArray(&Sorted_Examples));
    read_testing_data(&ds, Dataset.meta, &Dataset, &Subset, &Sorted_Examples, &Args);
    late_process_opts(Dataset.meta.num_attributes, Dataset.meta.num_examples, &Args);
    read_ensemble(&Ensemble, -1, 0, &Args);
}

START_TEST(check_LL_push)
{
    Prox_Matrix *list = NULL;
    LL_push(&list, 0, 0, 0.0);
    LL_push(&list, 1, 1, 1.0);
    LL_push(&list, 2, 2, 2.0);
    fail_unless(LL_length(list) == 3, "linked list is not the correct length");
    
    Prox_Matrix *current;
    current = list;
    fail_unless(current->row == 0 && current->col == 0 && av_eqf(current->data, 0.0), "first node is incorrect");
    current = current->next;
    fail_unless(current->row == 1 && current->col == 1 && av_eqf(current->data, 1.0), "second node is incorrect");
    current = current->next;
    fail_unless(current->row == 2 && current->col == 2 && av_eqf(current->data, 2.0), "third node is incorrect");
    current = current->next;
    fail_unless(current == NULL, "last node points to the wrong spot");
}
END_TEST

START_TEST(check_LL_unshift)
{
    Prox_Matrix *list = NULL;
    LL_unshift(&list, 0, 0, 0.0);
    LL_unshift(&list, 1, 1, 1.0);
    LL_unshift(&list, 2, 2, 2.0);
    fail_unless(LL_length(list) == 3, "linked list is not the correct length");
    
    Prox_Matrix *current;
    current = list;
    fail_unless(current->row == 2 && current->col == 2 && av_eqf(current->data, 2.0), "first node is incorrect");
    current = current->next;
    fail_unless(current->row == 1 && current->col == 1 && av_eqf(current->data, 1.0), "second node is incorrect");
    current = current->next;
    fail_unless(current->row == 0 && current->col == 0 && av_eqf(current->data, 0.0), "third node is incorrect");
    current = current->next;
    fail_unless(current == NULL, "last node points to the wrong spot");
}
END_TEST

START_TEST(check_node_matrix)
{
    int i;
    int **node_matrix;
    const int num_trees = 5;
    char message[1024] = {0};
    int truth_node_matrix[][5] = {
        {12,  3, 10,  2,  8},
        {11, 10, 30, 36, 30},
        {20, 10, 30, 36, 29},
        { 2,  2,  3,  2,  4},
        {18,  3,  8, 12, 20},
        {20,  9, 29, 30, 29},
        {20,  8, 30, 30, 28},
        { 5,  7, 29,  7, 27},
        {16,  3, 10, 10, 18},
        {20, 10, 30, 36, 30}
    };

    _read_data_and_trees();
    init_node_matrix(&Ensemble, &Subset, &node_matrix);
    for (i = 0; i < 10; ++i) {
        sprintf(message, "nodes for example %d are incorrect", i);
        fail_unless(! memcmp(truth_node_matrix[i], node_matrix[i], num_trees*sizeof(int)),  message);
    }
}
END_TEST

START_TEST(check_prox_matrix)
{
    int **node_matrix;
    Prox_Matrix *prox_matrix = NULL;
    Prox_Matrix *c;
    _read_data_and_trees();
    init_node_matrix(&Ensemble, &Subset, &node_matrix);
    assemble_prox_matrix(Subset.meta.num_examples, Ensemble.num_trees, node_matrix, &prox_matrix, FALSE);
    c = prox_matrix;
    fail_unless(c->row == 9 && c->col == 9 && av_eqf(c->data, 1.0), "element 9,9 is incorrect");
    c = c->next;
    fail_unless(c->row == 8 && c->col == 8 && av_eqf(c->data, 1.0), "element 8,8 is incorrect");
    c = c->next;
    fail_unless(c->row == 7 && c->col == 7 && av_eqf(c->data, 1.0), "element 7,7 is incorrect");
    c = c->next;
    fail_unless(c->row == 6 && c->col == 9 && av_eqf(c->data, 0.4), "element 6,9 is incorrect");
    c = c->next;
    fail_unless(c->row == 6 && c->col == 6 && av_eqf(c->data, 1.0), "element 6,6 is incorrect");
    c = c->next;
    fail_unless(c->row == 5 && c->col == 9 && av_eqf(c->data, 0.2), "element 5,9 is incorrect");
    c = c->next;
    fail_unless(c->row == 5 && c->col == 7 && av_eqf(c->data, 0.2), "element 5,7 is incorrect");
    c = c->next;
    fail_unless(c->row == 5 && c->col == 6 && av_eqf(c->data, 0.4), "element 5,6 is incorrect");
    c = c->next;
    fail_unless(c->row == 5 && c->col == 5 && av_eqf(c->data, 1.0), "element 5,5 is incorrect");
    c = c->next;
    fail_unless(c->row == 4 && c->col == 8 && av_eqf(c->data, 0.2), "element 4,8 is incorrect");
    c = c->next;
    fail_unless(c->row == 4 && c->col == 4 && av_eqf(c->data, 1.0), "element 4,4 is incorrect");
    c = c->next;
    fail_unless(c->row == 3 && c->col == 3 && av_eqf(c->data, 1.0), "element 3,3 is incorrect");
    c = c->next;
    fail_unless(c->row == 2 && c->col == 9 && av_eqf(c->data, 0.8), "element 2,9 is incorrect");
    c = c->next;
    fail_unless(c->row == 2 && c->col == 6 && av_eqf(c->data, 0.4), "element 2,6 is incorrect");
    c = c->next;
    fail_unless(c->row == 2 && c->col == 5 && av_eqf(c->data, 0.4), "element 2,5 is incorrect");
    c = c->next;
    fail_unless(c->row == 2 && c->col == 2 && av_eqf(c->data, 1.0), "element 2,2 is incorrect");
    c = c->next;
    fail_unless(c->row == 1 && c->col == 9 && av_eqf(c->data, 0.8), "element 1,9 is incorrect");
    c = c->next;
    fail_unless(c->row == 1 && c->col == 6 && av_eqf(c->data, 0.2), "element 1,6 is incorrect");
    c = c->next;
    fail_unless(c->row == 1 && c->col == 2 && av_eqf(c->data, 0.6), "element 1,2 is incorrect");
    c = c->next;
    fail_unless(c->row == 1 && c->col == 1 && av_eqf(c->data, 1.0), "element 1,1 is incorrect");
    c = c->next;
    fail_unless(c->row == 0 && c->col == 8 && av_eqf(c->data, 0.4), "element 0,8 is incorrect");
    c = c->next;
    fail_unless(c->row == 0 && c->col == 4 && av_eqf(c->data, 0.2), "element 0,4 is incorrect");
    c = c->next;
    fail_unless(c->row == 0 && c->col == 3 && av_eqf(c->data, 0.2), "element 0,3 is incorrect");
    c = c->next;
    fail_unless(c->row == 0 && c->col == 0 && av_eqf(c->data, 1.0), "element 0,0 is incorrect");
    fail_unless(c->next == NULL, "proximity matrix too big");
    
    // Reverse it
    Prox_Matrix *rev_prox_matrix = NULL;
    LL_reverse(prox_matrix, &rev_prox_matrix);
    c = rev_prox_matrix;
    fail_unless(c->row == 0 && c->col == 0 && av_eqf(c->data, 1.0), "element 0,0 is incorrect");
    c = c->next;
    fail_unless(c->row == 0 && c->col == 3 && av_eqf(c->data, 0.2), "element 0,3 is incorrect");
    c = c->next;
    fail_unless(c->row == 0 && c->col == 4 && av_eqf(c->data, 0.2), "element 0,4 is incorrect");
    c = c->next;
    fail_unless(c->row == 0 && c->col == 8 && av_eqf(c->data, 0.4), "element 0,8 is incorrect");
    c = c->next;
    fail_unless(c->row == 1 && c->col == 1 && av_eqf(c->data, 1.0), "element 1,1 is incorrect");
    c = c->next;
    fail_unless(c->row == 1 && c->col == 2 && av_eqf(c->data, 0.6), "element 1,2 is incorrect");
    c = c->next;
    fail_unless(c->row == 1 && c->col == 6 && av_eqf(c->data, 0.2), "element 1,6 is incorrect");
    c = c->next;
    fail_unless(c->row == 1 && c->col == 9 && av_eqf(c->data, 0.8), "element 1,9 is incorrect");
    c = c->next;
    fail_unless(c->row == 2 && c->col == 2 && av_eqf(c->data, 1.0), "element 2,2 is incorrect");
    c = c->next;
    fail_unless(c->row == 2 && c->col == 5 && av_eqf(c->data, 0.4), "element 2,5 is incorrect");
    c = c->next;
    fail_unless(c->row == 2 && c->col == 6 && av_eqf(c->data, 0.4), "element 2,6 is incorrect");
    c = c->next;
    fail_unless(c->row == 2 && c->col == 9 && av_eqf(c->data, 0.8), "element 2,9 is incorrect");
    c = c->next;
    fail_unless(c->row == 3 && c->col == 3 && av_eqf(c->data, 1.0), "element 3,3 is incorrect");
    c = c->next;
    fail_unless(c->row == 4 && c->col == 4 && av_eqf(c->data, 1.0), "element 4,4 is incorrect");
    c = c->next;
    fail_unless(c->row == 4 && c->col == 8 && av_eqf(c->data, 0.2), "element 4,8 is incorrect");
    c = c->next;
    fail_unless(c->row == 5 && c->col == 5 && av_eqf(c->data, 1.0), "element 5,5 is incorrect");
    c = c->next;
    fail_unless(c->row == 5 && c->col == 6 && av_eqf(c->data, 0.4), "element 5,6 is incorrect");
    c = c->next;
    fail_unless(c->row == 5 && c->col == 7 && av_eqf(c->data, 0.2), "element 5,7 is incorrect");
    c = c->next;
    fail_unless(c->row == 5 && c->col == 9 && av_eqf(c->data, 0.2), "element 5,9 is incorrect");
    c = c->next;
    fail_unless(c->row == 6 && c->col == 6 && av_eqf(c->data, 1.0), "element 6,6 is incorrect");
    c = c->next;
    fail_unless(c->row == 6 && c->col == 9 && av_eqf(c->data, 0.4), "element 6,9 is incorrect");
    c = c->next;
    fail_unless(c->row == 7 && c->col == 7 && av_eqf(c->data, 1.0), "element 7,7 is incorrect");
    c = c->next;
    fail_unless(c->row == 8 && c->col == 8 && av_eqf(c->data, 1.0), "element 8,8 is incorrect");
    c = c->next;
    fail_unless(c->row == 9 && c->col == 9 && av_eqf(c->data, 1.0), "element 9,9 is incorrect");
    fail_unless(c->next == NULL, "reverse matrix too big");
}
END_TEST

Suite *proximity_suite(void)
{
    Suite *suite = suite_create("Proximity");
    
    TCase *tc_proximity = tcase_create(" Check Proximity ");
    
    suite_add_tcase(suite, tc_proximity);
    tcase_add_test(tc_proximity, check_LL_push);
    tcase_add_test(tc_proximity, check_LL_unshift);
    tcase_add_test(tc_proximity, check_node_matrix);
    tcase_add_test(tc_proximity, check_prox_matrix);
        
    return suite;
}
