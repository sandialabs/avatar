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
/**
 * \file checkccregions.c
 * \brief Unit tests for regions module.
 *
 * $Source: /usr/local/Repositories/avatar/avatar/unittest/checkccregions.c,v $
 * $Revision: 1.2 $ 
 * $Date: 2006/02/03 06:10:50 $
 *
 * \modifications
 *    10-FEB-2006 KAB, added tests for checking max values
 */

#include <stdlib.h>
#include <string.h>
#include <check.h>
#include "fc.h"
#include "regions.h"
#include "checkall.h"

#define REGIONS_DATA "test_data.regions"
#define REGIONS_PRINT "test_print.regions"


// *********************************************
// ***** General library interface tests
// *********************************************

START_TEST(region_read) {
    FILE *file;
    timestep *timesteps;
    region_file_metadata metadata;
    FC_ReturnCode rc;

    // read
    file = fopen(REGIONS_DATA, "r");
    fail_unless( file != NULL, "failed to open test region data file for read");
    rc = read_region_data(file, &timesteps, &metadata);
    fclose(file);
    fail_unless( rc == FC_SUCCESS, "failed to read test region data");
    
    // compare
    fail_unless( ! compare_region_data(timesteps, metadata, timesteps, metadata), "region compare failed" );
    
    free_region_data(timesteps, metadata);
}
END_TEST

START_TEST(region_clone) {
    FILE *file;
    timestep *timesteps, *dup_timesteps;
    region_file_metadata metadata, dup_metadata;
    FC_ReturnCode rc;

    // read
    file = fopen(REGIONS_DATA, "r");
    fail_unless( file != NULL, "failed to open test region data file for read");
    rc = read_region_data(file, &timesteps, &metadata);
    fclose(file);
    fail_unless( rc == FC_SUCCESS, "failed to read test region data");

    // clone
    clone_region_data(timesteps, metadata, &dup_timesteps, &dup_metadata);
    fail_unless( ! compare_region_data(timesteps, metadata, dup_timesteps, dup_metadata), "region clone failed" );

    free_region_data(timesteps, metadata);
    free_region_data(dup_timesteps, dup_metadata);
}
END_TEST

START_TEST(region_print) {
    FILE *file;
    timestep *timesteps, *dup_timesteps;
    region_file_metadata metadata, dup_metadata;
    FC_ReturnCode rc;

    // read
    file = fopen(REGIONS_DATA, "r");
    fail_unless( file != NULL, "failed to open test region data file for read");
    rc = read_region_data(file, &timesteps, &metadata);
    fclose(file);
    fail_unless( rc == FC_SUCCESS, "failed to read test region data");

    // print
    file = fopen(REGIONS_PRINT, "w");
    fail_unless( file != NULL, "failed to open test region data file for print");
    print_region_data(timesteps, metadata, file);
    fclose(file);
    file = fopen(REGIONS_PRINT, "r");
    fail_unless( file != NULL, "failed to open test region data file for read");
    rc = read_region_data(file, &dup_timesteps, &dup_metadata);
    fclose(file);
    fail_unless( rc == FC_SUCCESS, "failed to read test region data");
    fail_unless( ! compare_region_data(timesteps, metadata, dup_timesteps, dup_metadata), "region clone failed" );
    
    free_region_data(timesteps, metadata);
    free_region_data(dup_timesteps, dup_metadata);
}
END_TEST

START_TEST(region_max) {
    FILE *file;
    timestep *timesteps;
    region_file_metadata metadata;
    FC_ReturnCode rc;

    // read
    file = fopen(REGIONS_DATA, "r");
    fail_unless( file != NULL, "failed to open test region data file for read");
    rc = read_region_data(file, &timesteps, &metadata);
    fclose(file);
    fail_unless( rc == FC_SUCCESS, "failed to read test region data");

    // get max values
    region_max_values max_vals = init_max_vals();
    fail_unless( max_vals.number == -1 && max_vals.label == -1 && max_vals.mesh == -1 && max_vals.size == -1,
                 "initialization of region_max_values struct failed" );
    get_region_data_max(timesteps, metadata, &max_vals);
    fail_unless( max_vals.number == 9 && max_vals.label == 9 && max_vals.mesh == 1 && max_vals.size == 3646,
                 "population of region_max_values struct failed" );
    max_vals.size = 5000;
    get_region_data_max(timesteps, metadata, &max_vals);
    fail_unless( max_vals.number == 9 && max_vals.label == 9 && max_vals.mesh == 1 && max_vals.size == 5000,
                 "re-population of region_max_values struct failed" );
    
    free_region_data(timesteps, metadata);

}
END_TEST

// *********************************************
// ***** Populate the Suite with the tests
// *********************************************

Suite *ccregions_suite(void)
{
    Suite *suite = suite_create("CCRegions");
    
    TCase *tc_ccregions = tcase_create(" - CCRegions Interface ");
    
    // general library init/final tests
    suite_add_tcase(suite, tc_ccregions);
    tcase_add_test(tc_ccregions, region_read);
    tcase_add_test(tc_ccregions, region_clone);
    tcase_add_test(tc_ccregions, region_print);
    tcase_add_test(tc_ccregions, region_max);
    
    return suite;
}
