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
#include "../src/util.h"
#include "../src/array.h"
#include "../src/distinct_values.h"


START_TEST(create_array)
{
    int i, value_count;
    float array[] = { 3, .31, 1.3, 2.3, 0.4, 3.9, 0.3, 1.3, 0.3, 4 };
    BST_Node *tree = NULL;
    Tree_Bookkeeping books = {0};
    books.num_malloced_nodes = 1;
    books.next_unused_node = 1;
    books.current_node = 0;
    tree = (BST_Node *)calloc(books.num_malloced_nodes, sizeof(BST_Node));
    
    tree[books.current_node].value = array[0];
    tree[books.current_node].left = -1;
    tree[books.current_node].right = -1;    
    value_count = 1;
    
    for (i = 1; i < 10; i++)
        value_count += tree_insert(&tree, &books, array[i]);
    fail_unless(value_count == 8, "should have 8 unique values");
    
    float truth[] = { 0.3, 0.31, 0.4, 1.3, 2.3, 3, 3.9, 4 };
    float *ordered_array;
    ordered_array = (float *)malloc(value_count * sizeof(float));
    tree_to_array(ordered_array, tree);
    fail_unless(! memcmp(ordered_array, truth, value_count * sizeof(float)), "ordered array not correct");
    
    free(ordered_array);
    free(tree);
}
END_TEST

START_TEST(check_translate_discrete)
{
    char *map[] = { "zero", "one", "two", "three", "four", "five", "six" };
    fail_unless(translate_discrete(map, 7, "neg-one") == -1, "didn't translate 'neg-one' to -1");
    fail_unless(translate_discrete(map, 7, "zero") == 0, "didn't translate 'zero' to 0");
    fail_unless(translate_discrete(map, 7, "one") == 1, "didn't translate 'one' to 1");
    fail_unless(translate_discrete(map, 7, "two") == 2, "didn't translate 'two' to 2");
    fail_unless(translate_discrete(map, 7, "three") == 3, "didn't translate 'three' to 3");
    fail_unless(translate_discrete(map, 7, "four") == 4, "didn't translate 'four' to 4");
    fail_unless(translate_discrete(map, 7, "five") == 5, "didn't translate 'five' to 5");
    fail_unless(translate_discrete(map, 7, "six") == 6, "didn't translate 'six' to 6");
    fail_unless(translate_discrete(map, 7, "seven") == -1, "didn't translate 'seven' to -1");
}
END_TEST

Suite *distinct_suite(void)
{
    Suite *suite = suite_create("DistinctValues");
    
    TCase *tc_create_array = tcase_create(" Create Array ");
    TCase *tc_translations = tcase_create(" Translations ");
    
    suite_add_tcase(suite, tc_create_array);
    tcase_add_test(tc_create_array, create_array);
    
    suite_add_tcase(suite, tc_translations);
    tcase_add_test(tc_create_array, check_translate_discrete);
    
    return suite;
}
