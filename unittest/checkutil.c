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
#include "../src/util.h"


START_TEST(test_factorial)
{
    //fail_unless(factorial(-4) == 1, "(-4)! should be 1");
    fail_unless(factorial(0) == 1, "0! should be 1");
    fail_unless(factorial(1) == 1, "1! should be 1");
    fail_unless(factorial(2) == 2, "2! should be 2");
    fail_unless(factorial(10) == 3628800, "10! should be 3628800");
}
END_TEST

START_TEST(test_exploding_filenames)
{
    char *filename;
    File_Bits bits;
    filename = strdup("/some/normal/path/to/some.file");
    bits = explode_filename(filename);
    fail_unless(! strcmp(bits.dirname,"/some/normal/path/to") &&
                ! strcmp(bits.basename,"some") &&
                ! strcmp(bits.extension,"file"), "Absolute path failed");
    filename = strdup("../../some/relative/path/to/some.file");
    bits = explode_filename(filename);
    fail_unless(! strcmp(bits.dirname,"../../some/relative/path/to") &&
                ! strcmp(bits.basename,"some") &&
                ! strcmp(bits.extension,"file"), "Relative path failed");
    filename = strdup("/some/normal/path/to/somefile");
    bits = explode_filename(filename);
    fail_unless(! strcmp(bits.dirname,"/some/normal/path/to") &&
                ! strcmp(bits.basename,"somefile") &&
                bits.extension == NULL, "Extensionless filename failed");
    filename = strdup("/some/normal/path/to/");
    bits = explode_filename(filename);
    fail_unless(! strcmp(bits.dirname,"/some/normal/path") &&
                ! strcmp(bits.basename,"to") &&
                bits.extension == NULL, "No filename failed");
    filename = strdup("/some/normal/path/to/double.dot.file");
    bits = explode_filename(filename);
    fail_unless(! strcmp(bits.dirname,"/some/normal/path/to") &&
                ! strcmp(bits.basename,"double.dot") &&
                ! strcmp(bits.extension,"file"), "Double dot filename failed");

}
END_TEST

START_TEST(test_num_digits)
{
    fail_unless(num_digits(0) == 1, "num_digits(0) s/b 1");
    fail_unless(num_digits(9) == 1, "num_digits(9) s/b 1");
    fail_unless(num_digits(10) == 2, "num_digits(10) s/b 2");
    fail_unless(num_digits(10001) == 5, "num_digits(10001) s/b 5");
    fail_unless(num_digits(-5) == 2, "num_digits(-5) s/b 2");
    fail_unless(num_digits(-10) == 3, "num_digits(-10) s/b 3");
}
END_TEST

Suite *util_suite(void)
{
    Suite *suite = suite_create("Utilities");
    
    TCase *tc_misc = tcase_create(" Miscellaneous ");
        
    suite_add_tcase(suite, tc_misc);
    tcase_add_test(tc_misc, test_factorial);
    tcase_add_test(tc_misc, test_exploding_filenames);
    tcase_add_test(tc_misc, test_num_digits);

    return suite;
}
