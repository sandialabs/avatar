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
#include <math.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/tree.h"

START_TEST(check_is_pure)
{
    int i;
    CV_Subset data = {0};
    data.meta.num_examples = 100;
    data.examples = (CV_Example *)malloc(data.meta.num_examples * sizeof(CV_Example));
    for (i = 0; i < data.meta.num_examples; i++)
        data.examples[i].containing_class_num = 0;
    fail_unless(is_pure(&data), "data should be labeled as 'pure'");
    data.examples[data.meta.num_examples - 1].containing_class_num = 1;
    fail_unless(! is_pure(&data), "data should be labeled as not 'pure' (1)");
    data.examples[data.meta.num_examples - 1].containing_class_num = 0;
    data.examples[data.meta.num_examples / 2].containing_class_num = 1;
    fail_unless(! is_pure(&data), "data should be labeled as not 'pure' (2)");
    data.examples[data.meta.num_examples / 2].containing_class_num = 0;
    data.examples[0].containing_class_num = 1;
    fail_unless(! is_pure(&data), "data should be labeled as not 'pure' (3)");
    data.meta.num_examples = 2;
    fail_unless(! is_pure(&data), "data should be labeled as not 'pure' (4)");
    data.meta.num_examples = 1;
    fail_unless(is_pure(&data), "data with one example should be, by default, 'pure'");
    
    free(data.examples);
}
END_TEST

START_TEST(check_find_best_class)
{
    int i;
    CV_Subset data = {0};
    data.meta.num_examples = 1000;
    data.meta.num_classes = 9;
    data.examples = (CV_Example *)malloc(data.meta.num_examples * sizeof(CV_Example));
    for (i = 0; i < data.meta.num_examples; i++)
        data.examples[i].containing_class_num = i % data.meta.num_classes;
    fail_unless(find_best_class(&data) == 0, "Failed to find class at beginning\n");
    fail_unless(errors_guessing_best_class(&data) == 888, "Wrong errors at beginning\n");
    data.examples[0].containing_class_num = data.meta.num_classes / 2;
    fail_unless(find_best_class(&data) == 4, "Failed to find best class in middle");
    fail_unless(errors_guessing_best_class(&data) == 888, "Wrong errors in middle\n");
    data.examples[0].containing_class_num = data.meta.num_classes - 1;
    fail_unless(find_best_class(&data) == 8, "Failed to find best class at end");
    fail_unless(errors_guessing_best_class(&data) == 888, "Wrong errors at end\n");
    
    free(data.examples);
}
END_TEST

Suite *tree_suite(void)
{
    Suite *suite = suite_create("Tree");
    
    TCase *tc_tree_utils = tcase_create(" Check TreeUtils ");
    suite_add_tcase(suite, tc_tree_utils);
    tcase_add_test(tc_tree_utils, check_is_pure);
    tcase_add_test(tc_tree_utils, check_find_best_class);
    
    return suite;
}
