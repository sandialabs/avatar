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
#include <stdio.h>
#include <string.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"

static Suite *dummy_suite(void)
{
  Suite *suite = suite_create("");
  return suite;
}

// Look for the string in a list of strings - returns index in list   
static int grep_string(char* query, int num, char list[][1024]) {
    int i;
    for (i = 0; i < num; i++) {
        if (!strcmp(query, list[i]))
            return i;
    }
    return -1; // didn't find it
}

void display_usage( void ) {
}

int main(int argc, char **argv) {

    int nf, i, indx;
    int num_run;
    int seen_one = 0;

    SRunner *srunner;
    enum print_output print_mode = CK_VERBOSE;
    
    char module_names[][1024] = {
	"api",
        "array",
        "bagging",
        "blearning",
        "boost",
        "crossval_util",
        "distinct",
        "diversity",
        "evaluate",
        "gain",
	"ivote",
        "knn",
        "majbagging",
	"missing",
        "proximity",
        "rwdata",
        "smote",
        "tree",
        "util",
    };
    int num_modules = sizeof(module_names) / sizeof(module_names[0]);
    char *run_names[num_modules];

    Suite *module_suites[] = {
	api_suite(),
        array_suite(),
        bagging_suite(),
        blearning_suite(),
        boost_suite(),
        crossval_util_suite(),
        distinct_suite(),
        diversity_suite(),
        eval_suite(),
        gain_suite(),
        ivote_suite(),
        knn_suite(),
        majbag_suite(),
	missing_suite(),
        proximity_suite(),
        rwdata_suite(),
        smote_suite(),
        tree_suite(),
        util_suite()
    };
    // handle arguments
    if (argc > 1 && (!strncasecmp(argv[1], "-h", sizeof(char)*2) || !strncasecmp(argv[1], "--h", sizeof(char)*3)) ) {
        printf("usage:\n");
        printf("   checkall               : runs unit tests for all modules with\n");
        printf("                            output only for failed tests\n");
        printf("   checkall module1 [...] : runs unit tests for only the requested\n");
        printf("                            modules with output for each test\n");
        printf("\n");
        printf("Possible modules:\n");
        for (i = 0; i < num_modules; i++)
            printf("   %s\n", module_names[i]);
        exit(0);
    } else {
        num_run = argc - 1;
        for (i = 1; i <= num_run; i++)
            run_names[i-1] = argv[i];
    }
    if (num_run == 0) {
        num_run = num_modules;
        for (i = 0; i < num_run; i++)
            run_names[i] = module_names[i];
    }
    
    // create srunner & add suites
    srunner = srunner_create(dummy_suite());
    for (i = 0; i < num_run; i++) {
        indx = grep_string(run_names[i], num_modules, module_names);
        if (indx > -1) {
            seen_one = 1;
            srunner_add_suite(srunner, module_suites[indx]);
        } else {
            printf("WARNING: No module named '%s' ... skipping\n", run_names[i]);
        }
    }
    if (! seen_one) {
        printf("ERROR: No valid modules given. Call with -h for list of modules\n");
        exit(-1);
    }
    
    // run the tests - Output type normal if doAll, verbose if not
    srunner_run_all (srunner, print_mode);
    nf = srunner_ntests_failed(srunner);

    // cleanup
    srunner_free(srunner);
    return (nf == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
