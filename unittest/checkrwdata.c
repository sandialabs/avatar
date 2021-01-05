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
#include <math.h>
#include "../src/crossval.h"
#include "check.h"
#include "checkall.h"
#include "../src/distinct_values.h"
#include "../src/rw_data.h"
#include "../src/schema.h"
#include "../src/tree.h"
#include "../src/options.h"


static CV_Subset Subset;
static DT_Ensemble Ensemble;
static Args_Opts Args;
static DT_Ensemble string_Ensemble;
static DT_Ensemble file_Ensemble;

#ifdef HAVE_AVATAR_FCLIB
START_TEST(load_dataset)
{
    int i, j;
    AV_ReturnCode rc;
    FC_Dataset dataset = {0};
    FC_Mesh mesh = {0};
    int mesh_dim = 2;
    double l_coords[] = { 0.0, 0.0, 0.0 };
    double u_coords[] = { 1.0, 1.0, 1.0 };
    FC_Sequence sequence = {0};
    double *coords_p;
    int num_steps = 2;
    FC_Variable **class_var;
    int num_class_steps;
    double class_var_data[num_steps][(mesh_dim+1)*(mesh_dim+1)*(mesh_dim+1)];
    FC_Variable **attribute_var;
    int num_attribute_steps;
    double attribute_var_data[num_steps][(mesh_dim+1)*(mesh_dim+1)*(mesh_dim+1)];
    CV_Class Class = {0};
    CV_Dataset Data = {0};
    
    // Initialize some stuff
	Class.class_var_name = strdup("class");
    Class.num_classes = 3;
    Class.thresholds = (float *)calloc(Class.num_classes - 1, sizeof(float));
    Class.thresholds[0] = 0.25;
    Class.thresholds[1] = 0.75;
    Class.class_frequencies = (int *)calloc(Class.num_classes, sizeof(int));
    Data.meta.num_fclib_seq = 0;
    Data.examples = (CV_Example *)malloc(sizeof(CV_Example));

    // Create dataset
    rc = fc_initLibrary();
    fail_unless(rc == AV_SUCCESS, "failed to initialize library");
    rc = fc_createDataset("test", &dataset);
    fail_unless(rc == AV_SUCCESS, "failed to create dataset");    
    // Add a simple 2x2x2 hex mesh
    rc = fc_createSimpleHexMesh(dataset, "mesh", mesh_dim, mesh_dim, mesh_dim, l_coords, u_coords, &mesh);
    fail_unless(rc == AV_SUCCESS, "failed to create mesh");
    // Create a sequence on the mesh
    rc = fc_createSequence(dataset, "sequence", &sequence);
    fail_unless(rc == AV_SUCCESS, "failed to create sequence");
    // Set sequence coordinates to the vertices
    rc = fc_getMeshCoordsPtr(mesh, &coords_p);
    fail_unless(rc == AV_SUCCESS, "failed to get coordinates pointer for mesh");
    rc = fc_setSequenceCoordsPtr(sequence, num_steps, FC_DT_DOUBLE, coords_p);
    fail_unless(rc == AV_SUCCESS, "failed to set sequence coordinates pointer");
    // Create the class variable on the sequence
    class_var = (FC_Variable **)malloc(sizeof(FC_Variable *));
    rc = fc_createSeqVariable(mesh, sequence, "class", &num_class_steps, class_var);
    fail_unless(rc == AV_SUCCESS, "failed to create class sequence variable");
    fail_unless(num_class_steps == num_steps, "class variable is not on all steps");
    // Create the attribute variable on the sequence
    attribute_var = (FC_Variable **)malloc(sizeof(FC_Variable *));
    rc = fc_createSeqVariable(mesh, sequence, "attribute", &num_attribute_steps, attribute_var);
    fail_unless(rc == AV_SUCCESS, "failed to create attribute sequence variable");
    fail_unless(num_attribute_steps == num_steps, "attribute variable is not on all steps");
    
    // Set variable data
    //
    // CLASS VARIABLE:
    // Timestep 0:  for z = 0.0 : data = 0.0000
    //              for z = 0.5 : data = 0.5000
    //              for z = 1.0 : data = 1.0000
    // Timestep 1:  for z = 0.0 : data = 0.0000
    //              for z = 0.5 : data = 0.3333...
    //              for z = 1.0 : data = 0.6666...
    //
    // ATTRIBUTE VARIABLE:
    // data = coordinate number + i
    
    for (i = 0; i < num_steps; i++) {
        for (j = 0; j < (mesh_dim+1)*(mesh_dim+1)*(mesh_dim+1); j++) {
            class_var_data[i][j] = (j / ((mesh_dim+1)*(mesh_dim+1))) / (2.0 + i);
            attribute_var_data[i][j] = (double)(j + i);
        }
        rc = fc_setVariableDataPtr(class_var[0][i], (mesh_dim+1)*(mesh_dim+1)*(mesh_dim+1), 1, FC_AT_VERTEX,
                                   FC_MT_SCALAR, FC_DT_DOUBLE, class_var_data[i]);
        fail_unless(rc == AV_SUCCESS, "failed to set data for class variable");
        rc = fc_setVariableDataPtr(attribute_var[0][i], (mesh_dim+1)*(mesh_dim+1)*(mesh_dim+1), 1, FC_AT_VERTEX,
                                   FC_MT_SCALAR, FC_DT_DOUBLE, attribute_var_data[i]);
        fail_unless(rc == AV_SUCCESS, "failed to set data for attribute variable");
    }
    fail_unless(fc_isSeqVariableValid(num_steps, class_var[0]), "class variable is not valid");
    fail_unless(fc_isSeqVariableValid(num_steps, attribute_var[0]), "attribute variable is not valid");
    
    // Add some data
    fail_unless(add_exo_data(dataset, 1, &Data, &Class, TRUE) == 1, "failed to add data for time 1");
    // Check that things are kosher
    fail_unless(
                 Data.meta.num_attributes == 1 && Data.meta.num_fclib_seq == 1 &&
                 Data.meta.num_examples_per_class[0] == 9 && Data.meta.num_examples_per_class[1] == 9 &&
                 Data.meta.num_examples_per_class[2] == 9 &&
                 Class.num_classes == 3 && Class.class_frequencies[0] == 9 &&
                 Class.class_frequencies[1] == 9 && Class.class_frequencies[2] == 9 &&
                 ! strcmp(Data.meta.attribute_names[0], "attribute"),
                 "failed to correctly read data for timestep 1"
               );
    
    // Add some more data
    fail_unless(add_exo_data(dataset, 2, &Data, &Class, FALSE) == 1, "failed to add data for time 2");
    // Check that things are still copasetic
    fail_unless(
                 Data.meta.num_attributes == 1 && Data.meta.num_fclib_seq == 2 &&
                 Data.meta.num_examples_per_class[0] == 18 && Data.meta.num_examples_per_class[1] == 27 &&
                 Data.meta.num_examples_per_class[2] == 9 &&
                 Class.num_classes == 3 && Class.class_frequencies[0] == 18 &&
                 Class.class_frequencies[1] == 27 && Class.class_frequencies[2] == 9 &&
                 ! strcmp(Data.meta.attribute_names[0], "attribute"),
                 "failed to correctly read data for timestep 2"
               );
               
    // Clean up
    free(Class.class_var_name);
    free(Class.thresholds);
    free(Class.class_frequencies);
    for (i = 0; i < Data.meta.num_attributes; i++)
        free(Data.meta.attribute_names[i]);
    free(Data.meta.attribute_names);
    free(Data.meta.attribute_types);
    free(Data.meta.global_offset);
    free(Data.meta.exo_data.seq_meshes);
    free(Data.examples);
    free(class_var);
    free(attribute_var);
    //fail_unless(fc_deleteSeqVariable(num_steps, class_var[0]) == AV_SUCCESS, "failed to delete class variable");
    //fail_unless(fc_deleteSeqVariable(num_steps, attribute_var[0]) == AV_SUCCESS, "failed to delete attribute variable");
    //fail_unless(fc_deleteSequence(sequence) == AV_SUCCESS, "failed to delete sequence");
    //fail_unless(fc_deleteMesh(mesh) == AV_SUCCESS, "failed to delete mesh");
    //fail_unless(fc_deleteDataset(dataset) == AV_SUCCESS, "failed to delete dataset");

}
END_TEST
#endif

START_TEST(names_file_test)
{
    Schema* file_schema =  read_schema("./data/proximity_test.names", 0, 4, 0, NULL);

    //Next, read all lines from .names file into a string, create another schema,
    //test if the two are equal

    FILE *fp;
    char str[10000];
    char* filename = "./data/proximity_test.names";

    char string_from_file[10000];
    
    fp = fopen(filename, "r");
    while (fgets(str,10000, fp) != NULL)
	strcat(string_from_file,str);

    fclose(fp);

    Schema* string_schema =  read_schema(string_from_file, 1, 4, 0, NULL);
    fail_unless(schema_num_attr(string_schema) == schema_num_attr(file_schema), "Schemas should be equal");
}
END_TEST


START_TEST(test_file_test)
{
    AV_SortedBlobArray file_Sorted_Examples;
    AV_SortedBlobArray string_Sorted_Examples;
    FC_Dataset file_ds = {0};
    FC_Dataset string_ds = {0};
    CV_Dataset file_Dataset = {0};
    CV_Dataset string_Dataset = {0};
    
    memset(&Args, 0, sizeof(Args_Opts));
    Args.caller = PROXIMITY_CALLER;
    Args.format = AVATAR_FORMAT;
    Args.base_filestem = strdup("proximity_test");

    Args.data_path = strdup("./data");
    Args.datafile = strdup("./data/proximity_test.data");
    Args.truth_column = 5;
    Args.save_trees = TRUE;

    memset(&Subset, 0, sizeof(CV_Subset));

    set_output_filenames(&Args, TRUE, TRUE);
    av_exitIfError(av_initSortedBlobArray(&file_Sorted_Examples));
    read_testing_data(&file_ds, file_Dataset.meta, &file_Dataset, &Subset, &file_Sorted_Examples, &Args);


    CV_Metadata file_meta = file_Dataset.meta;


    FILE *fp;
    char str[10000];
    char* filename = "./data/proximity_test.test";

    char string_from_file[10000];
    
    fp = fopen(filename, "r");
    while (fgets(str,10000, fp) != NULL)
	strcat(string_from_file,str);

    fclose(fp);

    Args.test_file_is_a_string = TRUE;
    Args.test_string = string_from_file;

    set_output_filenames(&Args, TRUE, TRUE);
    av_exitIfError(av_initSortedBlobArray(&string_Sorted_Examples));
    read_testing_data(&string_ds, string_Dataset.meta, &string_Dataset, &Subset, &string_Sorted_Examples, &Args);   

    CV_Metadata string_meta = string_Dataset.meta;

    fail_unless(file_meta.num_classes == string_meta.num_classes, "Dataset metas should have same number of classes");
    fail_unless(file_meta.num_fclib_seq == string_meta.num_fclib_seq, "Dataset metas should have the same number of fclib sequences");
    fail_unless(file_meta.num_examples == string_meta.num_examples, "Dataset metas should have the same number of examples");
    fail_unless(file_meta.num_attributes == string_meta.num_attributes, "Dataset metas should have the same number of attributes");
}
END_TEST


START_TEST(trees_file_test)
{
    AV_SortedBlobArray Sorted_Examples;
    FC_Dataset ds = {0};
    CV_Dataset Dataset = {0};
    //These global structs were left with dangling pointers by a previous test
    memset(&Subset, 0, sizeof(CV_Subset));
    memset(&Args, 0, sizeof(Args_Opts));
    memset(&string_Ensemble, 0, sizeof(DT_Ensemble));
    memset(&file_Ensemble, 0, sizeof(DT_Ensemble));
    
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
    read_ensemble(&file_Ensemble, -1, 0, &Args);

    char* filename = "./data/proximity_test.trees";
    FILE* fp = fopen(filename, "r");
    fseek(fp, 0, SEEK_END);
    size_t fileLen = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* string_from_file = malloc(fileLen + 1);
    fread(string_from_file, 1, fileLen, fp);
    string_from_file[fileLen] = 0;
    fclose(fp);

    Args.trees_file_is_a_string = TRUE;
    Args.trees_string = string_from_file;
    
    memset(&Dataset, 0, sizeof(CV_Dataset));

    set_output_filenames(&Args, TRUE, TRUE);
    av_exitIfError(av_initSortedBlobArray(&Sorted_Examples));
    read_testing_data(&ds, Dataset.meta, &Dataset, &Subset, &Sorted_Examples, &Args);
    late_process_opts(Dataset.meta.num_attributes, Dataset.meta.num_examples, &Args);
    read_ensemble(&string_Ensemble, -1, 0, &Args);

    fail_unless(file_Ensemble.num_classes == string_Ensemble.num_classes, "Ensembles should have same number of classes");
    fail_unless(file_Ensemble.num_trees == string_Ensemble.num_trees, "Ensembles should have same number of trees");
    fail_unless(file_Ensemble.num_training_examples == string_Ensemble.num_training_examples, "Ensembles should have same number of training examples");
    fail_unless(file_Ensemble.num_attributes == string_Ensemble.num_attributes, "Ensembles should have same number of attributes");

}
END_TEST


Suite *rwdata_suite(void)
{
    Suite *suite = suite_create("Read/Write_Data");
    
    TCase *tc_loaddataset = tcase_create(" Load Dataset ");
    TCase *tc_stringinput = tcase_create(" String Input ");

    suite_add_tcase(suite, tc_loaddataset);
#ifdef HAVE_AVATAR_FCLIB
    tcase_add_test(tc_loaddataset, load_dataset);
#endif

    suite_add_tcase(suite, tc_stringinput);
    tcase_add_test(tc_stringinput, names_file_test);
    tcase_add_test(tc_stringinput, test_file_test);
    tcase_add_test(tc_stringinput, trees_file_test);
    
    return suite;
}
