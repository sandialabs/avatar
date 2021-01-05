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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include "crossval.h"
#include "distinct_values.h"
#include "rw_data.h"
#include "tree.h"
#include "util.h"
#include "evaluate.h"
#include "array.h"
#include "options.h"
#include "missing_values.h"
#include "memory.h"
#include "safe_memory.h"
#include "schema.h"

#include <execinfo.h>

// Forward declarations of prototypes for helper functions.
void update_skipped_features(int num_attributes, Args_Opts *args);
void _show_skipped_features(FILE* fh, const char* msg, int num_skipped, const int* skipped);

void read_metadata(FC_Dataset *ds, CV_Metadata *meta, Args_Opts *args) {
    int i;
    CV_Class Class;
    
    // Initialize some stuff
    if (args->format == EXODUS_FORMAT) {
        #ifdef HAVE_AVATAR_FCLIB
        // Read .classes file for exodus datasets
        read_classes_file(&Class, args);
        // Let command line -V option override class file or set args value if not set
        if (args->class_var_name != NULL)
            Class.class_var_name = av_strdup(args->class_var_name);
        else
            args->class_var_name = av_strdup(Class.class_var_name);
        meta->num_classes =  Class.num_classes;
        free(meta->class_names);
        meta->class_names = (char **)malloc(Class.num_classes * sizeof(char *));
        for (i = 0; i < Class.num_classes; i++)
            meta->class_names[i] = av_strdup(Class.class_names[i]);
        meta->num_fclib_seq = 0;
        #else
        av_missingFCLIB();
        #endif
    }
    meta->exo_data.num_seq_meshes = 0;
    
    if (args->format == EXODUS_FORMAT) {
        #ifdef HAVE_AVATAR_FCLIB
        // Add the training data for the requested timesteps
        //printf("Trying to open '%s'\n", args->datafile);
        for (i = 0; i < args->num_train_times; i++) {
            //printf("Adding training data for timestep %d\n", args->train_times[i]);
            if (! add_exo_metadata(*ds, args->train_times[i], meta, &Class)) {
                fprintf(stderr, "Error adding data for timestep %d\n", args->train_times[i]);
                exit(-8);
            }
        }
        #endif
    } else {
        if (! read_names_file(meta, &Class, args, TRUE)) {
            fprintf(stderr, "Error reading names file\n");
            exit(-8);
        }
    }
    
    free_CV_Class(Class);
}

void read_testing_data(FC_Dataset *ds, CV_Metadata train_meta, CV_Dataset *dataset, CV_Subset *subset, AV_SortedBlobArray *sorted_examples, Args_Opts *args) {
    int i;
    CV_Class Class = {0};
    AV_ReturnCode rc;
    
    // Initialize some stuff
    if (args->format == EXODUS_FORMAT) {
        // Read .classes file for exodus datasets
        read_classes_file(&Class, args);
        // Let command line -V option override class file or set args value if not set
        if (args->class_var_name != NULL)
            Class.class_var_name = av_strdup(args->class_var_name);
        else
            args->class_var_name = av_strdup(Class.class_var_name);
        dataset->meta.num_classes =  Class.num_classes;
        free(dataset->meta.class_names);
        dataset->meta.class_names = (char **)malloc(Class.num_classes * sizeof(char *));
        for (i = 0; i < Class.num_classes; i++)
            dataset->meta.class_names[i] = av_strdup(Class.class_names[i]);
        dataset->meta.num_fclib_seq = 0;
    }
    dataset->meta.exo_data.num_seq_meshes = 0;
    dataset->examples = (CV_Example *)calloc(1, sizeof(CV_Example));
    rc = (AV_ReturnCode) av_initSortedBlobArray(sorted_examples);
    av_exitIfError(rc);

    if (args->do_training) {
        // Need the missing values from the training data
        subset->meta.Missing = train_meta.Missing;
    }
    
    if (args->format == EXODUS_FORMAT) {
        // Add the testing data for the requested timesteps
        for (i = 0; i < args->num_test_times; i++) {
            //printf("Adding testing data for timestep %d\n", args->test_times[i]);
            if (! add_exo_data(*ds, args->test_times[i], dataset, &Class, (i==0?TRUE:FALSE))) {
                fprintf(stderr, "Error adding data for timestep %d\n", args->test_times[i]);
                exit(-8);
            }
        }
    } else if (args->format == AVATAR_FORMAT) {
        if (! read_names_file(&dataset->meta, &Class, args, (args->do_training == TRUE ? FALSE : TRUE))) {
            fprintf(stderr, "Error reading names file\n");
            exit(-8);
        } 
        if (! read_data_file(dataset, subset, &Class, sorted_examples, "test", *args)) {
            fprintf(stderr, "Error reading data file\n");
            exit(-8);
        }
        subset->meta.exo_data.num_seq_meshes = 0;
    }
    
    // The functionality of this block is done in read_data_file() for non-exodus data
    if (args->format == EXODUS_FORMAT) {
        // Create the testing CV_Subset
        create_cv_subset(*dataset, subset);
        
        //printf("%d testing sequences were added\n", subset->num_fclib_seq);
        //printf("Global offsets:");
        //for (i = 0; i <= subset->num_fclib_seq; i++)
        //    printf(" %d", subset->global_offset[i]);
        //printf("\n");
        
        // Put data into SortedBlobArray and populate distinct_attribute_values in each example
        populate_distinct_values_from_dataset(*dataset, subset, sorted_examples);
        //printf("There are %d examples in the testing dataset\n", Test_Sorted_Examples.numBlob);
    }
}

void read_training_data(FC_Dataset *ds, CV_Dataset *dataset, CV_Subset *subset, AV_SortedBlobArray *sorted_examples, Args_Opts *args) {
    int i;
    CV_Class Class;
    AV_ReturnCode rc;
    
    // Initialize some stuff
    if (args->format == EXODUS_FORMAT) {
        // Read .classes file for exodus datasets
        read_classes_file(&Class, args);
        // Let command line -V option override class file or set args value if not set
        if (args->class_var_name != NULL)
            Class.class_var_name = av_strdup(args->class_var_name);
        else
            args->class_var_name = av_strdup(Class.class_var_name);
        dataset->meta.num_classes =  Class.num_classes;
        free(dataset->meta.class_names);
        dataset->meta.class_names = (char **)malloc(Class.num_classes * sizeof(char *));
        for (i = 0; i < Class.num_classes; i++)
            dataset->meta.class_names[i] = av_strdup(Class.class_names[i]);
        dataset->meta.num_fclib_seq = 0;
    }
    dataset->meta.exo_data.num_seq_meshes = 0;
    dataset->examples = (CV_Example *)malloc(sizeof(CV_Example));
    rc = av_initSortedBlobArray(sorted_examples);
    av_exitIfError(rc);
    
    if (args->format == EXODUS_FORMAT) {
        // Add the training data for the requested timesteps
        //printf("Trying to open '%s'\n", args->datafile);
        for (i = 0; i < args->num_train_times; i++) {
            //printf("Adding training data for timestep %d\n", args->train_times[i]);
            if (! add_exo_data(*ds, args->train_times[i], dataset, &Class, (i==0?TRUE:FALSE))) {
                fprintf(stderr, "Error adding data for timestep %d\n", args->train_times[i]);
                exit(-8);
            }
        }
    } else if (args->format == AVATAR_FORMAT) {
        if (! read_names_file(&dataset->meta, &Class, args, TRUE)) {
            fprintf(stderr, "Error reading names file\n");
            exit(-8);
        }
        if (! read_data_file(dataset, subset, &Class, sorted_examples, "data", *args)) {
            fprintf(stderr, "Error reading data file\n");
            exit(-8);
        }

        subset->meta.exo_data.num_seq_meshes = 0;
        //printf("Read %d features for each of %d samples\n", subset.num_attributes, subset.num_examples);
    }
    
    // The functionality of this block is done in read_data_file() for non-exodus data
    if (args->format == EXODUS_FORMAT) {
        // Create the integer-mapped arrays for each attribute
        create_cv_subset(*dataset, subset);
        
        //printf("%d training sequences were added\n", subset.num_fclib_seq);
        //printf("Global offsets:");
        //for (i = 0; i <= subset.num_fclib_seq; i++)
        //    printf(" %d", subset.global_offset[i]);
        //printf("\n");
        
        // Put data into SortedBlobArray and populate distinct_attribute_values in each example
        populate_distinct_values_from_dataset(*dataset, subset, sorted_examples);
        //printf("There are %d examples in the training dataset\n", sorted_examples.numBlob);
    }
    
    // free_CV_Class destroys dataset.meta.class_names so it's not accessible in the calling function.
    // Not sure why, exactly
    //free_CV_Class(Class);
}

void init_fc(Args_Opts args) {
    #ifdef HAVE_AVATAR_FCLIB
    // init library and load dataset
    av_exitIfErrorPrintf((AV_ReturnCode) fc_setLibraryVerbosity(args.verbosity), "Failed to set verbosity to %d\n", args.verbosity);
    av_exitIfErrorPrintf((AV_ReturnCode) fc_initLibrary(), "Could not initLibrary\n");
    #else
    av_missingFCLIB();
    #endif
}

void open_exo_datafile(FC_Dataset *ds, char *datafile) {
    #ifdef HAVE_AVATAR_FCLIB
    av_exitIfErrorPrintf((AV_ReturnCode) fc_loadDataset(datafile, ds), "Could not load dataset: '%s'\n", datafile);
    #else
    av_missingFCLIB();
    #endif
}

int add_exo_data(FC_Dataset ds, int timestep, CV_Dataset *data, CV_Class *class, Boolean init) {//, BST_Node ***tree) {
    #ifdef HAVE_AVATAR_FCLIB
    int gid;
    FC_ReturnCode rc;
    int num_seq_vars, num_meshes;
    int *num_steps_per_var;
    int num_data_points;
    int class_var_num = -1;
    int att_num;
    FC_Mesh *seq_meshes;
    FC_Variable **seq_vars;
    char **seq_var_names;
    FC_AssociationType class_var_type, var_type;
    
    // Get the meshes for this timestep
    rc = fc_getNumMesh(ds, &num_meshes);
    av_exitIfError((AV_ReturnCode) rc);
    //printf("exo dataset has %d meshes\n", num_meshes);
    rc = fc_getMeshes(ds, &num_meshes, &seq_meshes);
    //printf("exo dataset has %d meshes\n", num_meshes);
    //printf("Mesh 0: %d %d\n", seq_meshes[0].slotID, seq_meshes[0].uID);
    av_exitIfError((AV_ReturnCode) rc);
    data->meta.exo_data.seq_meshes = (FC_Mesh *)malloc(num_meshes * sizeof(FC_Mesh));
    if (init == TRUE)
        data->meta.num_examples_per_class = (int *)calloc(class->num_classes, sizeof(int));
    
    data->meta.exo_data.num_seq_meshes = num_meshes;
    
    // For each mesh ...
    for (int i = 0; i < num_meshes; i++) {
        // Store mesh
        data->meta.exo_data.seq_meshes[i] = seq_meshes[i];
        
        // Update number of sequences
        data->meta.num_fclib_seq++;
        
        // ... get sequence variables
        rc = fc_getSeqVariables(seq_meshes[i], &num_seq_vars, &num_steps_per_var, &seq_vars);
        seq_var_names = (char **)malloc(num_seq_vars * sizeof(char *));
        av_exitIfError((AV_ReturnCode) rc);
        // ... check that the class variable is one of them and get it's type
        int found_class_var = 0;
        for (int j = 0; j < num_seq_vars; j++) {
            rc = fc_getVariableName(seq_vars[j][timestep - 1], &seq_var_names[j]);
            av_exitIfErrorPrintf((AV_ReturnCode) rc, "Failed to get name for variable %d on mesh %d\n", j, i);
            if (! strcmp(seq_var_names[j], class->class_var_name)) {
                fc_getVariableAssociationType(seq_vars[j][timestep - 1], &class_var_type);
                found_class_var = 1;
                class_var_num = j;
                data->meta.exo_data.assoc_type = class_var_type;
            }
        }
        if (! found_class_var)
            return 0;
        
        // ... count all other variables of same type as class var
        att_num = 0;
        for (int j = 0; j < num_seq_vars; j++) {
            // skip the class var
            if (j == class_var_num)
                continue;
            fc_getVariableAssociationType(seq_vars[j][timestep - 1], &var_type);
            if (var_type != class_var_type)
                continue;
            // Count this as a valid attribute
            att_num++;
        }
        
        // ... make sure we have the same number of attributes or set the number if it's our first time through
        // ... malloc or realloc as necessary
        if (data->meta.num_fclib_seq == 1) {
            data->meta.num_attributes = att_num;
            data->meta.attribute_names = (char **)malloc(data->meta.num_attributes * sizeof(char *));
            data->meta.attribute_types = (Attribute_Type *)malloc(data->meta.num_attributes * sizeof(Attribute_Type));
            free(data->meta.global_offset);
            data->meta.global_offset = (int *)calloc(2, sizeof(int));
            free(data->meta.exo_data.variables);
            data->meta.exo_data.variables = (FC_Variable **)malloc(sizeof(FC_Variable *));
            data->meta.exo_data.variables[0] = (FC_Variable *)malloc(data->meta.num_attributes * sizeof(FC_Variable));
            //data->data_ptr = (void ***)malloc(data->meta.num_attributes * sizeof(void **));
            //for (j = 0; j < data->meta.num_attributes; j++) {
            //    data->data_ptr[j] = (void **)malloc(sizeof(void *));
            //}
        } else {
            if (att_num != data->meta.num_attributes) {
                fprintf(stderr, "Found %d attributes on mesh %d for timestep %d but expected %d\n",
                                att_num, i, timestep, data->meta.num_attributes);
                return 0;
            }
            data->meta.global_offset = (int *)realloc(data->meta.global_offset, (data->meta.num_fclib_seq + 1) * sizeof(int));
            data->meta.exo_data.variables = (FC_Variable **)realloc(data->meta.exo_data.variables,
                                                                   data->meta.num_fclib_seq * sizeof(FC_Variable *));
            data->meta.exo_data.variables[data->meta.num_fclib_seq - 1] =
                                                    (FC_Variable *)malloc(data->meta.num_attributes * sizeof(FC_Variable));
            //for (j = 0; j < data->meta.num_attributes; j++) {
            //    data->data_ptr[j] = (void **)realloc(data->data_ptr[j], data->meta.num_fclib_seq * sizeof(void *));
            //}
        }
        
        // Get data for class and store attribute FC_Variable pointers
        att_num = 0;
        for (int j = 0; j < num_seq_vars; j++) {
            fc_getVariableAssociationType(seq_vars[j][timestep - 1], &var_type);
            if (var_type != class_var_type)
                continue;

            if (j == class_var_num) {
                void *class_data_ptr;
                rc = fc_getVariableDataPtr(seq_vars[j][timestep - 1], &class_data_ptr);
                av_exitIfErrorPrintf((AV_ReturnCode) rc, "Failed top get data pointer for class var %s on mesh %d at timestep %d\n",
                                         seq_var_names[j], i, timestep);

                // Also update global_offset
                rc = fc_getVariableNumDataPoint(seq_vars[j][timestep - 1], &num_data_points);
                av_exitIfErrorPrintf((AV_ReturnCode) rc, "Failed to get number of data points for %s on mesh %d at timestep %d\n",
                                         class->class_var_name, i, timestep);
                data->meta.global_offset[data->meta.num_fclib_seq] =
                                                        data->meta.global_offset[data->meta.num_fclib_seq - 1] + num_data_points;
                data->meta.num_examples = data->meta.global_offset[data->meta.num_fclib_seq];
                // Also update CV_Example array
                data->examples = (CV_Example *)realloc(data->examples, data->meta.num_examples * sizeof(CV_Example));
                if (! data->examples)
                    return 0;
                
                for (int k = 0; k < num_data_points; k++) {
                    gid = fclib2global(data->meta.num_fclib_seq - 1, k, data->meta.num_fclib_seq, data->meta.global_offset);
                    data->examples[gid].fclib_seq_num = data->meta.num_fclib_seq - 1;
                    data->examples[gid].fclib_id_num = k;
                    data->examples[gid].global_id_num = gid;
                    data->examples[gid].containing_class_num = get_class_index(*((double *)class_data_ptr + k), class);
                    data->meta.num_examples_per_class[data->examples[gid].containing_class_num]++;
                    data->examples[gid].predicted_class_num = -1;
                    class->class_frequencies[data->examples[gid].containing_class_num]++;
                }
                
            } else {
                if (data->meta.num_fclib_seq == 1) {
                    data->meta.attribute_names[att_num] = av_strdup(seq_var_names[j]);
                    data->meta.attribute_types[att_num] = CONTINUOUS;
                }
                //printf("Setting exo_data.variables[%d][%d]\n", data->meta.num_fclib_seq - 1, att_num);
                data->meta.exo_data.variables[data->meta.num_fclib_seq - 1][att_num] = seq_vars[j][timestep - 1];
                // Release all timesteps not being used
                for (int k = 0; k < num_steps_per_var[j]; k++) {
                    if (k != timestep - 1) {
                        rc = fc_releaseVariable(seq_vars[j][k]);
                        if (rc != FC_SUCCESS)
                            fprintf(stderr, "Failed to release variable %d at timestep %d: %d\n", j, k, rc);
                    }
                }
                att_num++;
            }
        }
        
        for (int j = 0; j < num_seq_vars; j++)
            free(seq_var_names[j]);
        free(seq_var_names);
        
    }
    
    return 1;
    #else
    av_missingFCLIB();
    return 0;
    #endif
}

int add_exo_metadata(FC_Dataset ds, int timestep, CV_Metadata *data, CV_Class *class) {//, BST_Node ***tree) {
    #ifdef HAVE_AVATAR_FCLIB
    int i, j, k;
    FC_ReturnCode rc;
    int num_seq_vars, num_meshes;
    int *num_steps_per_var;
    int num_data_points;
    int class_var_num = -1;
    int att_num;
    FC_Mesh *seq_meshes;
    FC_Variable **seq_vars;
    char **seq_var_names;
    FC_AssociationType class_var_type, var_type;
    
    // Get the meshes for this timestep
    rc = fc_getNumMesh(ds, &num_meshes);
    av_exitIfError((AV_ReturnCode) rc);
    //printf("exo dataset has %d meshes\n", num_meshes);
    rc = fc_getMeshes(ds, &num_meshes, &seq_meshes);
    //printf("exo dataset has %d meshes\n", num_meshes);
    //printf("Mesh 0: %d %d\n", seq_meshes[0].slotID, seq_meshes[0].uID);
    av_exitIfError((AV_ReturnCode) rc);
    data->exo_data.seq_meshes = (FC_Mesh *)malloc(num_meshes * sizeof(FC_Mesh));
    data->exo_data.num_seq_meshes = num_meshes;
    
    // For each mesh ...
    for (i = 0; i < num_meshes; i++) {
        // Store mesh
        data->exo_data.seq_meshes[i] = seq_meshes[i];
        
        // Update number of sequences
        data->num_fclib_seq++;
        
        // ... get sequence variables
        rc = fc_getSeqVariables(seq_meshes[i], &num_seq_vars, &num_steps_per_var, &seq_vars);
        seq_var_names = (char **)malloc(num_seq_vars * sizeof(char *));
        av_exitIfError((AV_ReturnCode) rc);
        // ... check that the class variable is one of them and get it's type
        int found_class_var = 0;
        for (j = 0; j < num_seq_vars; j++) {
            rc = fc_getVariableName(seq_vars[j][timestep - 1], &seq_var_names[j]);
            av_exitIfErrorPrintf((AV_ReturnCode) rc, "Failed to get name for variable %d on mesh %d\n", j, i);
            if (! strcmp(seq_var_names[j], class->class_var_name)) {
                fc_getVariableAssociationType(seq_vars[j][timestep - 1], &class_var_type);
                found_class_var = 1;
                class_var_num = j;
                data->exo_data.assoc_type = class_var_type;
            }
        }
        if (! found_class_var)
            return 0;
        
        // ... count all other variables of same type as class var
        att_num = 0;
        for (j = 0; j < num_seq_vars; j++) {
            // skip the class var
            if (j == class_var_num)
                continue;
            fc_getVariableAssociationType(seq_vars[j][timestep - 1], &var_type);
            if (var_type != class_var_type)
                continue;
            // Count this as a valid attribute
            att_num++;
        }
        
        // ... make sure we have the same number of attributes or set the number if it's our first time through
        // ... malloc or realloc as necessary
        if (data->num_fclib_seq == 1) {
            data->num_attributes = att_num;
            data->attribute_names = (char **)malloc(data->num_attributes * sizeof(char *));
            data->attribute_types = (Attribute_Type *)malloc(data->num_attributes * sizeof(Attribute_Type));
            free(data->global_offset);
            data->global_offset = (int *)calloc(2, sizeof(int));
            free(data->exo_data.variables);
            data->exo_data.variables = (FC_Variable **)malloc(sizeof(FC_Variable *));
            data->exo_data.variables[0] = (FC_Variable *)malloc(data->num_attributes * sizeof(FC_Variable));
            //data->data_ptr = (void ***)malloc(data->num_attributes * sizeof(void **));
            //for (j = 0; j < data->num_attributes; j++) {
            //    data->data_ptr[j] = (void **)malloc(sizeof(void *));
            //}
        } else {
            if (att_num != data->num_attributes) {
                fprintf(stderr, "Found %d attributes on mesh %d for timestep %d but expected %d\n",
                                att_num, i, timestep, data->num_attributes);
                return 0;
            }
            data->global_offset = (int *)realloc(data->global_offset, (data->num_fclib_seq + 1) * sizeof(int));
            data->exo_data.variables = (FC_Variable **)realloc(data->exo_data.variables,
                                                                   data->num_fclib_seq * sizeof(FC_Variable *));
            data->exo_data.variables[data->num_fclib_seq - 1] =
                                                    (FC_Variable *)malloc(data->num_attributes * sizeof(FC_Variable));
            //for (j = 0; j < data->num_attributes; j++) {
            //    data->data_ptr[j] = (void **)realloc(data->data_ptr[j], data->num_fclib_seq * sizeof(void *));
            //}
        }
        
        // Get data for class and store attribute FC_Variable pointers
        att_num = 0;
        for (j = 0; j < num_seq_vars; j++) {
            fc_getVariableAssociationType(seq_vars[j][timestep - 1], &var_type);
            if (var_type != class_var_type)
                continue;

            if (j == class_var_num) {
                void *class_data_ptr;
                rc = fc_getVariableDataPtr(seq_vars[j][timestep - 1], &class_data_ptr);
                av_exitIfErrorPrintf((AV_ReturnCode) rc, "Failed top get data pointer for class var %s on mesh %d at timestep %d\n",
                                         seq_var_names[j], i, timestep);

                // Also update global_offset
                rc = fc_getVariableNumDataPoint(seq_vars[j][timestep - 1], &num_data_points);
                av_exitIfErrorPrintf((AV_ReturnCode) rc, "Failed to get number of data points for %s on mesh %d at timestep %d\n",
                                         class->class_var_name, i, timestep);
                data->global_offset[data->num_fclib_seq] =
                                                        data->global_offset[data->num_fclib_seq - 1] + num_data_points;
                
            } else {
                if (data->num_fclib_seq == 1) {
                    data->attribute_names[att_num] = av_strdup(seq_var_names[j]);
                    data->attribute_types[att_num] = CONTINUOUS;
                }
                //printf("Setting exo_data.variables[%d][%d]\n", data->num_fclib_seq - 1, att_num);
                data->exo_data.variables[data->num_fclib_seq - 1][att_num] = seq_vars[j][timestep - 1];
                // Release all timesteps not being used
                for (k = 0; k < num_steps_per_var[j]; k++) {
                    if (k != timestep - 1) {
                        rc = fc_releaseVariable(seq_vars[j][k]);
                        if ((AV_ReturnCode) rc != AV_SUCCESS)
                            fprintf(stderr, "Failed to release variable %d at timestep %d: %d\n", j, k, rc);
                    }
                }
                att_num++;
            }
        }
        
        for (j = 0; j < num_seq_vars; j++)
            free(seq_var_names[j]);
        free(seq_var_names);
        
    }
    
    return 1;
    #else
    av_missingFCLIB();
    return 0;
    #endif
}

/*
 * Not currently used
 *
int add_fold_data(int num_folds, CV_Dataset dataset, CV_Subset *subset, int **fold_pop, Args_Opts args) {
    int i;
    FILE *fh;
    char strbuf[262144];
    int att_num, fold_num;
    int num_examples;
    int malloc_size = 1000;
    BST_Node **tree;
    Tree_Bookkeeping *books;
    
    // This malloc's the initial examples array for subset to ex_malloc_size
    // REMEMBER that the examples array for dataset is not malloc'ed
    copy_dataset_meta(dataset, subset, malloc_size);

    // First time through, we need to create the distinct value mapping
    
    // Initialize one tree and one set of high/low for each attribute
    tree = (BST_Node **)malloc(subset->meta.num_attributes * sizeof(BST_Node *));
    books = (Tree_Bookkeeping *)malloc(subset->meta.num_attributes * sizeof(Tree_Bookkeeping));
    for (i = 0; i < subset->meta.num_attributes; i++) {
        books[i].num_malloced_nodes = 1000;
        books[i].next_unused_node = 1;
        books[i].current_node = 0;
        tree[i] = (BST_Node *)malloc(books[i].num_malloced_nodes * sizeof(BST_Node));
    }
    subset->low = (int *)malloc(subset->meta.num_attributes * sizeof(int));
    subset->high = (int *)malloc(subset->meta.num_attributes * sizeof(int));
    *fold_pop = (int *)calloc(num_folds, sizeof(int));
    
    for (fold_num = 0; fold_num < num_folds; fold_num++) {

        if ((fh = fopen(args.test_file, "r")) == NULL) {
            fprintf(stderr, "Failed to open *.test file: '%s'\n", args.test_file);
            return 0;
        }
        
        att_num = 0;
        while (fscanf(fh, "%s", strbuf) > 0) {
            // Each line has data.num_attributes attribute values and then the class value
            if (att_num == 0) {
                // This is the start of a line so create a new CV_Example
                subset->meta.num_examples++;
                if (subset->meta.num_examples > malloc_size) {
                    malloc_size *= 2;
                    subset->examples = (CV_Example *)realloc(subset->examples, malloc_size * sizeof(CV_Example));
                }
            }

            if (att_num == dataset.meta.num_attributes) {
                // This value is the class label
                subset->examples[subset->meta.num_examples-1].containing_class_num = atoi(strbuf);
                subset->examples[subset->meta.num_examples-1].predicted_class_num = -1;
                // Might as well set fold number and alloc d_a_v, too
                (*fold_pop)[fold_num]++;
                subset->examples[subset->meta.num_examples-1].containing_fold_num = fold_num;
                subset->examples[subset->meta.num_examples-1].distinct_attribute_values =
                                                                (int *)malloc(subset->meta.num_attributes * sizeof(int));
                // Reset att_num
                att_num = 0;
            } else {
                // Handle attribute value
                if (dataset.meta.attribute_types[att_num] == CONTINUOUS) {
                    // Handle a continuous attribute
                    if (books[att_num].current_node == 0) {
                        tree[att_num][0].value = atof(strbuf);
                        tree[att_num][0].left = -1;
                        tree[att_num][0].right = -1;
                        subset->low[att_num] = 0;
                        subset->high[att_num] = 0;
                        
                        // current_node == 0 only triggers the initialization. Otherwise we don't care about it.
                        // Increment it only so the initialization is not repeated
                        books[att_num].current_node++;
                    } else {
                        subset->high[att_num] += tree_insert(&tree[att_num], &books[att_num], atof(strbuf));
                    }
                } else {
                    printf("Woops -- shouldn't have gotten here: %d\n", dataset.meta.attribute_types[att_num]);
                }
                
                att_num++;
            }
        }
        
        fclose(fh);
    }
    
    // Create the float array to translate int back to float for each attribute
    dataset.meta.num_examples = subset->meta.num_examples;
    subset->float_data = (float **)malloc(subset->meta.num_attributes * sizeof(float *));
    for (i = 0; i < subset->meta.num_attributes; i++) {
        subset->float_data[i] = (float *)malloc((subset->high[i] + 1)*sizeof(float));
        tree_to_array(subset->float_data[i], tree[i]);
        free(tree[i]);
    }
    free(tree);
    subset->examples = (CV_Example *)realloc(subset->examples, subset->meta.num_examples * sizeof(CV_Example));
    
    // Second time through, the distinct values for each attribute at each example are added
    
    num_examples = 0;
    for (fold_num = 0; fold_num < num_folds; fold_num++) {

        if ((fh = fopen(args.test_file, "r")) == NULL) {
            fprintf(stderr, "Failed to open *.test file: '%s'\n", args.test_file);
            return 0;
        }
        
        att_num = 0;
        while (fscanf(fh, "%s", strbuf) > 0) {
            if (att_num == 0)
                num_examples++;
            
            if (att_num == dataset.meta.num_attributes) {
                // Reset att_num
                att_num = 0;
            } else {
                // Handle attribute value
                if (dataset.meta.attribute_types[att_num] == CONTINUOUS) {
                    // Handle a continuous attribute
                    subset->examples[num_examples-1].distinct_attribute_values[att_num] =
                                    translate(subset->float_data[att_num], atof(strbuf), 0, subset->high[att_num] + 1);
                } else {
                    printf("Woops -- shouldn't have gotten here: %d\n", dataset.meta.attribute_types[att_num]);
                }
                
                att_num++;
            }
        }
        
        fclose(fh);
    }

    return 1;
}
 *
 */

//NOTE: since args is passed by value, any pointer members allocated here must be freed at the end of the function
//Use argsIn to remember the pointers that were passed in, so that it's easy to tell which ones changed in args locally
int read_data_file(CV_Dataset *data, CV_Subset *sub, CV_Class *class, AV_SortedBlobArray *blob, char *ext, Args_Opts args) {
    Args_Opts argsIn = args;
    int i, j, k, all_atts;
    char *filename = NULL;
    FILE *fh = NULL;
    //char strbuf[262144];
    char *strbuf;
    int num_elements;
    char **elements;
    AV_ReturnCode rc;
    
    // Set up stuff for filling in missing values
    int disc_count, cont_count; // Keep separate track of which discrete and which continuous att we're on
                                // Both start at 0 and go to num_discrete_atts and num_continuous_atts, respectively
    int *num_continuous_exs;    // Array holding number of examples with non-missing values for continuous attribute
    int *num_discrete_exs;      // Array holding number of examples with non-missing values for discrete attribute
    char *this_att;             // The current attribute value's character representation
    float **continuous_values;  // Array holding continuous att values for each continuous att and each example
    int **discrete_values;      // Array holding discrete att value index for each discrete att and each example
    int sizeof_cont_disc_val = 1000;    // Current size of continuous_values and discrete_values
    int num_continuous_atts = 0;
    int num_discrete_atts = 0;
    for (i = 0; i < data->meta.num_attributes; i++) {
        if (data->meta.attribute_types[i] == DISCRETE)
            num_discrete_atts++;
        else if (data->meta.attribute_types[i] == CONTINUOUS)
            num_continuous_atts++;
    }
    num_continuous_exs = (int *)calloc(num_continuous_atts, sizeof(int));
    continuous_values = (float **)malloc(num_continuous_atts * sizeof(float *));
    for (i = 0; i < num_continuous_atts; i++)
        continuous_values[i] = (float *)malloc(sizeof_cont_disc_val * sizeof(float));
    num_discrete_exs = (int *)calloc(num_discrete_atts, sizeof(int));
    discrete_values = (int **)malloc(num_discrete_atts * sizeof(int *));
    for (i = 0; i < num_discrete_atts; i++)
        discrete_values[i] = (int *)malloc(sizeof_cont_disc_val * sizeof(int));
    
    sub->meta.num_classes = data->meta.num_classes;
    sub->meta.num_attributes = data->meta.num_attributes;
    sub->meta.class_names = data->meta.class_names;
    sub->meta.attribute_names = data->meta.attribute_names;
    sub->meta.attribute_types = data->meta.attribute_types;
    sub->meta.num_discrete_values = data->meta.num_discrete_values;
    sub->meta.discrete_attribute_map = data->meta.discrete_attribute_map;
    sub->meta.exo_data.num_seq_meshes = data->meta.exo_data.num_seq_meshes;
    free(sub->meta.num_examples_per_class);
    sub->meta.num_examples_per_class = (int *)calloc(class->num_classes, sizeof(int));
    data->meta.num_examples = 0;
    
    // Allocate one tree and one set of high/low for each attribute
    init_att_handling(sub->meta);
    
    if (! strcmp(ext, "data")){
	filename = av_strdup(args.datafile);
        if (args.train_file_is_a_string == TRUE){
	    fh = fmemopen(args.train_string, strlen(args.train_string), "r");
    	}
    	else {
            if ((fh = fopen(filename, "r")) == NULL) {
                fprintf(stderr, "Failed to open file \"%s\" for reading.\n", filename);
                return 0;
    	    }
        }
    }
    else if (! strcmp(ext, "test")){
        filename = av_strdup(args.test_file);
    	if (args.test_file_is_a_string == TRUE){
	    fh = fmemopen(args.test_string, strlen(args.test_string), "r");
    	}
    	else {
            if ((fh = fopen(filename, "r")) == NULL) {
                fprintf(stderr, "Failed to open file \"%s\" for reading.\n", filename);
                return 0;
    	    }
       }
    }
    
    //while (fscanf(fh, "%[^\n]", strbuf) > 0 || fscanf(fh, "%[\n]", strbuf) > 0) {
    while (read_line(fh, &strbuf) > 0) {
        //if (strbuf[0] == '\n')
        //    continue;
        
        strip_lt_whitespace(strbuf);
        if (*strbuf == '\0')
            continue;

        //parse_comma_sep_string(strbuf, &num_elements, &elements);
        parse_delimited_string(',', strbuf, &num_elements, &elements);
        //printf("Got %d elements processing '%s'\n",num_elements, strbuf);
        free(strbuf);

        if (elements[0][0] == '#') {
            // Look and process labels
            int found_label = 0;
            // Split elements[0] on spaces
            int num_tkns;
            char **tkns;
            //printf("Splitting '%s' on spaces\n", elements[0]);
            parse_space_sep_string(elements[0], &num_tkns, &tkns);
            //printf("Got %d tokens\n", num_tkns);
            // Look for 'labels' in first or second tkns -- depending on whether it's '#labels' or '# labels'
            if (num_tkns > 0 && strlen(tkns[0]) > 6) {
                char *junk;
                junk = tkns[0] + strlen(tkns[0]) - 6;
                if (! strcasecmp(junk, "labels"))
                    found_label = 1;
            } else if (num_tkns > 1 && ! strcasecmp(tkns[1], "labels")) {
                found_label = 2;
            }
	    //	    printf("tkns[0] = %s\n",tkns[0]);
	    //	    printf("found_label = %d\n",found_label);
            if (found_label > 0){// && found_label < num_tkns) {
                // Process attribute labels
                
                // First, replace elements[0] with first label (i.e. remove '#labels') so elements holds the labels
                char *temp;
                temp = (char *)malloc(strlen(elements[0]) * sizeof(char));
                temp = strstr(elements[0], tkns[found_label]);
                elements[0] = av_strdup(temp);
                // printf("Got %d attributes, %d skipped features, %d elements\n", sub->meta.num_attributes, args.num_skipped_features, num_elements);
                if (sub->meta.num_attributes + args.num_skipped_features + 1 != num_elements) { // +1 for the class column
                    fprintf(stderr, "ERROR: .%s file has %d columns but should have %d\n",
                                     ext, num_elements, sub->meta.num_attributes+args.num_skipped_features+1);
                    exit(-1);
                }
                int num_mismatches = 0;
                int used_atts = -1;
                // Count number of mismatched attribute names
                for ( k = 0; k < num_elements; k++) {
                    //printf("Looking at element %d. Current att_num = %d\n", k+1, used_atts+1);
                    // Skip over skipped features
                  if (find_int(k+1, args.num_skipped_features, args.skipped_features)) {
                        //printf("Skipping feature in column %d\n", k+1);
                        continue;
                    }
                    // Skip over truth column
                    if (args.truth_column == k+1) {
                        //printf("Skipping class in column %d\n", k+1);
                        continue;
                    }
                    used_atts++;
                    //printf("Using attribute %d = %s from %s file as internal att %d\n", k+1, elements[k], ext, used_atts);
                    //printf("Comparing column %d from data file with entry %d from names file\n", k+1, used_atts+1);
                    if (strcmp(elements[k], sub->meta.attribute_names[used_atts])) {
                        // Try ignoring a leading number and compare the rest. data_inspector adds a column number
                        // in the names file which causes problems.
                        char *this_label;
                        int this_label_num = strtol(sub->meta.attribute_names[used_atts], &this_label, 10);
                        // The number should correspond to column number and the non-leading-number portion
                        // (after skipping one space) should be the attribute label
                        if (strcmp(elements[k], this_label+1) || this_label_num != k+1) {
                            num_mismatches++;
                            fprintf(stderr, "ERROR: attribute in column %d has conflicting names in .%s file: '%s' and '%s'\n",
                                            k+1, ext, elements[k], sub->meta.attribute_names[used_atts]);
                        }
                    }
                }
                if (num_mismatches > 0)
                    exit(-1);
            }
            
            for (k = 0; k < num_tkns; k++)
                free(tkns[k]);
            free(tkns);
            for (k = 0; k < num_elements; k++)
                free(elements[k]);
            free(elements);
            
            // Ignore comments
            continue;
        }

        // Make sure we have enough elements for all attributes (including skipped ones) and the true class
        if (num_elements < data->meta.num_attributes + args.num_skipped_features + 1){
            fprintf(stderr, "Expected at least %d+%d+1 values but read %d ... skipping\n",
                                                        data->meta.num_attributes, args.num_skipped_features, num_elements);
            continue;
        }
        
        // We now have a valid example ...
        data->meta.num_examples++;
        j = -1; // This is the index for the current attribute allowing for skips
        disc_count = 0;
        cont_count = 0;
        
        // Make sure arrays of values for missing value computation are big enough
        while (data->meta.num_examples > sizeof_cont_disc_val) {
            sizeof_cont_disc_val *= 2;
            for (i = 0; i < num_continuous_atts; i++)
                continuous_values[i] = (float *)realloc(continuous_values[i], sizeof_cont_disc_val * sizeof(float));
            for (i = 0; i < num_discrete_atts; i++)
                discrete_values[i] = (int *)realloc(discrete_values[i], sizeof_cont_disc_val * sizeof(int));
        }
        
        // Build tree for getting distinct values for continuous attributes
        // Also, temporarily store data for filling in missing values
        
        // Loop over all features but we'll skip args.num_skipped_features to be left with data.num_attributes
        for (all_atts = 0; all_atts < data->meta.num_attributes + args.num_skipped_features; all_atts++) {
            if (args.truth_column > 0 && all_atts+1 >= args.truth_column &&
                find_int(all_atts+2, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is in a column past the truth column so look for the
                // attribute number + 2 (extra +1 is because all_atts is 0-based) in the list
                // of features to skip since the list is based on column number and not attribute number
                //printf("Skipping this att because %d+1 is in the list of features to skip\n", all_atts+1);
                continue;
            }
            if (args.truth_column > 0 && all_atts+1 < args.truth_column &&
                find_int(all_atts+1, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is before the truth column so attribute number corresponds
                // to column number.
                //printf("Skipping this att because %d is in the list of features to skip\n", all_atts+1);
                continue;
            }
            if (args.truth_column < 0 &&
                find_int(all_atts+1, args.num_skipped_features, args.skipped_features)) {
                // The truth column is last so all attributes are before the truth column
                //printf("Skipping this att because %d is in the list of features to skip\n", all_atts+1);
                continue;
            }
            
            //printf("Processing feature %d\n", all_atts);
            j++; // This is the index for the current attribute allowing for skips
            //printf("Global (1-based) feature %d is this run's (1-based) feature %d\n", all_atts+1, j+1);
            
            //printf("Using column %d for this att\n", all_atts + (all_atts < args.truth_column-1 ? 0 : 1));
            this_att = av_strdup(elements[all_atts + (all_atts < args.truth_column-1 ? 0 : 1)]);
            //printf("Read '%s' as att value\n", this_att);
            int dv = process_attribute_char_value(this_att, j, sub->meta);
            /*
            if(dv == -1)
            {
              fprintf(stderr, "Failed to find attribute \"%s\".\n", this_att);
              return 0;
            }
            */
            if (sub->meta.attribute_types[j] == DISCRETE) {
                if (strcmp(this_att, "?")) {
                    discrete_values[disc_count][num_discrete_exs[disc_count]++] = dv;
                }
                disc_count++;
            } else if (sub->meta.attribute_types[j] == CONTINUOUS) {
                if (strcmp(this_att, "?")) {
                    // Add continous attribute value to array for missing value computation
                    continuous_values[cont_count][num_continuous_exs[cont_count]++] = atof(this_att);
                }
                cont_count++;
            }
            free(this_att);
        }

        for (i = 0; i < num_elements; i++)
            free(elements[i]);
        free(elements);

    }

    if (data->meta.num_examples == 0) {
        fprintf(stderr, "No examples found in the .%s file: '%s'\n", ext, filename);
	return 0;
    }
    
    fclose(fh);
    
    data->examples = (CV_Example *)realloc(data->examples, data->meta.num_examples * sizeof(CV_Example));
    if (!data->meta.global_offset)
    {
      data->meta.global_offset = (int *)calloc(2, sizeof(int));
    }
    data->meta.global_offset[1] = data->meta.num_examples;
    sub->meta.global_offset = data->meta.global_offset;
    sub->examples = data->examples;
    sub->meta.num_examples = data->meta.num_examples;
    
    // Compute missing values if this is training data
    if (! strcmp(ext, "data")) {
        calculate_missing_values(continuous_values, num_continuous_exs, discrete_values, num_discrete_exs, sub);
    // If it is testing data, use the missing values from the training data or read in the ensemble file
    // But add it to the tree to make sure it's there
    } else {
        // If we are training, then we've read the training file already and know the missing values
        // so use them from the CV_Subset
        if (args.do_training == TRUE) {
            for (i = 0; i < sub->meta.num_attributes; i++)
                if (sub->meta.attribute_types[i] == CONTINUOUS)
                    process_attribute_float_value(sub->meta.Missing[i].Continuous, i);
        // If we didn't train, then we don't know the missing values yet.
        // In this case, use a throw-away ensemble to read the metadata in the trees file to get them
        } else {
            if (args.partitions_filename != NULL) {
                // If we are using a partition file, then add all missing values from all partitions' ensemble files
                CV_Partition partitions;
                read_partition_file(&partitions, &args);
                // Save some parameters
                char *df = av_strdup(args.datafile);
                char *bf = av_strdup(args.base_filestem);
                char *dp = av_strdup(args.data_path);
                free(args.datafile);
                free(args.base_filestem);
                free(args.data_path);
                // Read all partition tree files
                for (i = 0; i < partitions.num_partitions; i++) {
                    args.datafile = av_strdup(partitions.partition_datafile[i]);
                    args.base_filestem = av_strdup(partitions.partition_base_filestem[i]);
                    args.data_path = av_strdup(partitions.partition_data_path[i]);
                    if (args.partitions_filename != NULL)
                        set_output_filenames(&args, TRUE, TRUE);
                    else
                        set_output_filenames(&args, FALSE, FALSE);
                    
                    DT_Ensemble junk_ensemble = {0};
                    FILE *tree_file;
                    
                    // Initialize
                    junk_ensemble.num_classes = 0;
                    junk_ensemble.num_attributes = 0;
                    junk_ensemble.num_trees = 0;
                    
                    // Open tree file
                    if (args.trees_file_is_a_string == TRUE){
                      tree_file = fmemopen(args.trees_string, strlen(args.trees_string), "r");
                    } else {
                      if ((tree_file = fopen(args.trees_file, "r")) == NULL) {
                        fprintf(stderr, "Failed to open file for reading trees: '%s'\nExiting ...\n", args.trees_file);
                        exit(8);
                      }
                    }
                    read_ensemble_metadata(tree_file, &junk_ensemble, 1, &args);
/* NEED TO FIGURE OUT WHAT MISSING VALUES TO USE WHEN DOING TESTING SINCE WE HAVE MULTIPLTE SETS */
                    for (i = 0; i < junk_ensemble.num_attributes; i++)
                        if (junk_ensemble.attribute_types[i] == CONTINUOUS)
                            process_attribute_float_value(junk_ensemble.Missing[i].Continuous, i);
                    fclose(tree_file);
                    free_DT_Ensemble(junk_ensemble, TEST_MODE);
                }
                // Restore
                args.datafile = av_strdup(df);
                args.base_filestem = av_strdup(bf);
                args.data_path = av_strdup(dp);
                free(df);
                free(bf);
                free(dp);
            } else {
                DT_Ensemble junk_ensemble = {0};
                char *tree_filename = build_output_filename(-1, args.trees_file, args);
                FILE *tree_file;
                
                // Initialize
                junk_ensemble.num_classes = 0;
                junk_ensemble.num_attributes = 0;
                junk_ensemble.num_trees = 0;
                
                // Open tree file
                if (args.trees_file_is_a_string == TRUE){
                  tree_file = fmemopen(args.trees_string, strlen(args.trees_string), "r");
                } else {
                  if ((tree_file = fopen(tree_filename, "r")) == NULL) {
                    fprintf(stderr, "Failed to open file for reading trees: '%s'\nExiting ...\n", tree_filename);
                    exit(8);
                  }
                }

                read_ensemble_metadata(tree_file, &junk_ensemble, 1, &args);
                free(sub->meta.Missing);
                sub->meta.Missing = (union data_point_union *)malloc(junk_ensemble.num_attributes * sizeof(union data_point_union));
                for (i = 0; i < junk_ensemble.num_attributes; i++) {
                    if (junk_ensemble.attribute_types[i] == CONTINUOUS) {
                        process_attribute_float_value(junk_ensemble.Missing[i].Continuous, i);
                        sub->meta.Missing[i].Continuous = junk_ensemble.Missing[i].Continuous;
                    } else if (junk_ensemble.attribute_types[i] == DISCRETE) {
                        sub->meta.Missing[i].Discrete = junk_ensemble.Missing[i].Discrete;
                    }
                }
                free(tree_filename);
                fclose(tree_file);
                free_DT_Ensemble(junk_ensemble, TEST_MODE);
            }
        }
    }
    
    // Create the float array to translate int back to float for each attribute
    create_float_data(sub);
    
    // Free up temp storage of values
    for (i = 0; i < num_continuous_atts; i++)
        free(continuous_values[i]);
    free(continuous_values);
    free(num_continuous_exs);
    for (i = 0; i < num_discrete_atts; i++)
        free(discrete_values[i]);
    free(discrete_values);
    free(num_discrete_exs);
    
    /*
     * Read data file again and populate the distinct_attribute_value array
     */

    if (args.test_file_is_a_string == TRUE){
	fh = fmemopen(args.test_string, strlen(args.test_string), "r");
    } else {
        if ((fh = fopen(filename, "r")) == NULL) {
                fprintf(stderr, "Failed to open .%s file: '%s'\n", ext, filename);
                return 0;
    	 }
    }
    
    
    
    int ex_num = -1;
    int num_class_xlate_errors = 0;
    int num_att_xlate_errors = 0;
    int line_num = 1;
    //while (fscanf(fh, "%[^\n]", strbuf) > 0 || fscanf(fh, "%[\n]", strbuf) > 0) {
    while (read_line(fh, &strbuf) > 0) {
      /*
      {
        char prefix[81];
        memcpy(prefix,strbuf,80);
        printf("reading line = %s\n",prefix);
      }
      */
        //if (strbuf[0] == '\n') {
        //    line_num++;
        //    continue;
        //}
        
        strip_lt_whitespace(strbuf);
        if (*strbuf == '\0')
            continue;

        //parse_comma_sep_string(strbuf, &num_elements, &elements);
        parse_delimited_string(',', strbuf, &num_elements, &elements);
        free(strbuf);

        // Skip comments
        if (elements[0][0] == '#') {
            for (i = 0; i < num_elements; i++)
                free(elements[i]);
            free(elements);
            continue;
        }

        // Make sure we have enough elements for all attributes and the true class
        if (num_elements < data->meta.num_attributes + args.num_skipped_features + 1) {
            fprintf(stderr, "Expected at least %d+%d+1 values but read %d ... skipping\n",
                                                data->meta.num_attributes, args.num_skipped_features, num_elements);
            for (i = 0; i < num_elements; i++)
                free(elements[i]);
            free(elements);
            continue;
        }
        
        // We now have a valid example ...
        ex_num++;
        j = -1; // This is the index for the current attribute allowing for skips

        // Populate the distinct_attribute_values array for each attribute
        // continuous attributes get the index into float_data
        // discrete attributes get the index of the discrete attribute value
        
        // Loop over all features but we'll skip args.num_skipped_features to be left with data.num_attributes
        //printf("Loopin over %d atts\n", data->meta.num_attributes + args.num_skipped_features);
        for (all_atts = 0; all_atts < data->meta.num_attributes + args.num_skipped_features; all_atts++) {
        //for (all_atts = 0; all_atts < data->meta.num_attributes; all_atts++) {
            
            if (args.truth_column > 0 && all_atts+1 >= args.truth_column &&
                find_int(all_atts+2, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is in a column past the truth column so look for the
                // attribute number + 2 (extra +1 is because all_atts is 0-based) in the list
                // of features to skip since the list is based on column number and not attribute number
                //printf("Skipping this att because %d+1 is in the list of features to skip\n", all_atts+1);
                continue;
            }
            if (args.truth_column > 0 && all_atts+1 < args.truth_column &&
                find_int(all_atts+1, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is before the truth column so attribute number corresponds
                // to column number.
                //printf("Skipping this att because %d is in the list of features to skip\n", all_atts+1);
                continue;
            }
            if (args.truth_column < 0 &&
                find_int(all_atts+1, args.num_skipped_features, args.skipped_features)) {
                // The truth column is last so all attributes are before the truth column
                //printf("Skipping this att because %d is in the list of features to skip\n", all_atts+1);
                continue;
            }
            j++;
            //printf("Processing global feature %d (%s) and this run's feature %d\n", all_atts, sub->meta.attribute_names[j], j);
            
            if (j == 0) { // This is the first attribute seen for this example so init some stuff
                sub->examples[ex_num].global_id_num = sub->examples[ex_num].fclib_id_num = ex_num;
                sub->examples[ex_num].fclib_seq_num = 0;
                sub->examples[ex_num].distinct_attribute_values = (int *)malloc(sub->meta.num_attributes * sizeof(int));
                rc = av_addBlobToSortedBlobArray(blob, &sub->examples[ex_num], cv_example_compare_by_seq_id);
                if (rc < 0) {
                    av_exitIfErrorPrintf(rc, "Failed to add example %d to SBA\n", ex_num);
                } else if (rc == 0) {
                    fprintf(stderr, "Example %d already exists in SBA\n", ex_num);
                }
            }
            this_att = av_strdup(elements[all_atts + (all_atts < args.truth_column-1 ? 0 : 1)]);


            num_att_xlate_errors += add_attribute_char_value(this_att, sub, ex_num, j, all_atts, line_num,
                                                             filename, args.truth_column, elements);
            free(this_att);
        }
        // Populate the class for this example
        // The class is in the last column
        sub->examples[ex_num].predicted_class_num = -1;
        //printf("Trying to translate a truth value of '%s' using", elements[args.truth_column-1]);
        //for (i = 0; i < sub->meta.num_classes; i++)
        //    printf(" '%s'", sub->meta.class_names[i]);
        //printf("\n");
        sub->examples[ex_num].containing_class_num =
                    translate_discrete(sub->meta.class_names, sub->meta.num_classes, elements[args.truth_column-1]);
        //printf("Incrementing nepc %d\n", sub->examples[ex_num].containing_class_num);
        //printf("  was %d\n", sub->meta.num_examples_per_class[sub->examples[ex_num].containing_class_num]);
        if (sub->examples[ex_num].containing_class_num < 0) {
            // This is an invalid class
            if (args.output_accuracies == ON) {
                num_class_xlate_errors++;
                fprintf(stderr, "Invalid class '%s': line %d of %s\n", elements[args.truth_column-1], line_num, filename);
            }
        } else {
            sub->meta.num_examples_per_class[sub->examples[ex_num].containing_class_num]++;
            class->class_frequencies[sub->examples[ex_num].containing_class_num]++;
        }
        //if (ex_num == 0)
        //    cv_example_print(sub->examples[ex_num], sub->meta.num_attributes);
        
        for (i = 0; i < num_elements; i++)
            free(elements[i]);
        free(elements);
        
        line_num++;
    }
    
    fclose(fh);
    free(filename);

    if(argsIn.skipped_features != args.skipped_features)
      free(args.skipped_features);

    find_int_release();

    if (num_class_xlate_errors > 0 || num_att_xlate_errors > 0)
        return 0;
    return 1;
}

/*
 * Not currently used
 *
int read_opendt_names_file(CV_Dataset *data, CV_Class *class, Args_Opts args) {
    FILE *fh;
    char strbuf[262144];
    char *prefix = "default_attribute_";
    
    data->meta.num_classes = class->num_classes;
    data->meta.num_attributes = 0;
    
    //printf("Reading metadata from %s\n", args.names_file);
    if ((fh = fopen(args.names_file, "r")) == NULL) {
        fprintf(stderr, "Failed to open .names file: '%s'\n", args.names_file);
        return 0;
    }
    
    while (fscanf(fh, "%s", strbuf) > 0) {
        if (! strcasecmp(strbuf, "continuous") || ! strcasecmp(strbuf, "discrete")) {
            // Handle attributes
            data->meta.num_attributes++;
            if (data->meta.num_attributes == 1) {
                data->meta.attribute_names = (char **)malloc(sizeof(char *));
                data->meta.attribute_types = (Attribute_Type *)malloc(sizeof(Attribute_Type));
            } else {
                data->meta.attribute_names = (char **)realloc(data->meta.attribute_names, data->meta.num_attributes * sizeof(char *));
                data->meta.attribute_types = (Attribute_Type *)realloc(data->meta.attribute_types,
                                                                  data->meta.num_attributes * sizeof(Attribute_Type));
            }
            if (! strcasecmp(strbuf, "continuous")) {
                data->meta.attribute_names[data->meta.num_attributes-1] =
                                                    (char *)malloc((num_digits(data->meta.num_attributes)+19) * sizeof(char));
                sprintf(data->meta.attribute_names[data->meta.num_attributes-1], "%s%d", prefix, data->meta.num_attributes);
                data->meta.attribute_types[data->meta.num_attributes-1] = CONTINUOUS;
            } else {
                printf("Woops -- shouldn't have gotten here\n");
            }
        } else {
            // Handle class names
        }
    }
    
    fclose(fh);
    
    return 1;
}
 *
 */

int read_names_file(CV_Metadata *meta, CV_Class *class, Args_Opts *args, Boolean update_skip_list) {
    int i;
    uint num_columns = 0;
    uint colID = 0;
    int manual_target = -1;
    int* manual_exclude = NULL;
    //printf("Reading metadata from %s\n", namesfile);
    //_show_skipped_features(stdout, "Initial skipped feature list: ", args->num_skipped_features, args->skipped_features);

    // Convert target and list of feature indices from 1-based to 0-based.
    manual_target = args->truth_column - 1;
    manual_exclude = NULL;
    if (args->num_skipped_features > 0) {
        manual_exclude = e_calloc(args->num_skipped_features, sizeof(int));
        for (i = 0; i < args->num_skipped_features; ++i) {
            manual_exclude[i] = args->skipped_features[i] - 1;
        }
    }

    // Read in the schema information.
    char* names_data;

    if (args->names_file_is_a_string == TRUE){
	names_data = args->names_string;
    } else {
	names_data = args->names_file;
    }

    Schema* schema = read_schema(
    names_data,
    args->names_file_is_a_string,
    manual_target,
    args->num_skipped_features,
    manual_exclude);

    if (NULL == schema) {
        free(manual_exclude);
        return 0;
    }
    //write_schema(stdout, schema);

    //
    // Fill in meta data structure.
    //

    // Cosmin added if statement (09/01/2020)
    if (meta->global_offset != NULL) 
        free(meta->global_offset);
    meta->global_offset = e_calloc(2, sizeof(int));

    // Count the number of active attributes (response variable excluded).
    meta->num_attributes = 0;
    num_columns = schema_num_attr(schema);
    for (colID = 0; colID < num_columns; ++colID) {
        Boolean isActivePredictor = schema_attr_is_active(schema, colID) 
            && schema_attr_type(schema, colID) != CLASS;
        if (isActivePredictor) {
            meta->num_attributes += 1;
        }
    }

    // Fill in the meta data for the active predictors.
    meta->attribute_names = e_calloc(meta->num_attributes, sizeof(char*));
    meta->attribute_types = e_calloc(meta->num_attributes, sizeof(Attribute_Type));
    meta->num_discrete_values = e_calloc(meta->num_attributes, sizeof(int));
    meta->discrete_attribute_map = e_calloc(meta->num_attributes, sizeof(char**));
    i = 0;
    for (colID = 0; colID < num_columns; ++colID) {
        Boolean isActivePredictor = schema_attr_is_active(schema, colID) 
            && schema_attr_type(schema, colID) != CLASS;
        if (isActivePredictor) {
            meta->attribute_names[i] = e_strdup(schema_attr_name(schema, colID));
            meta->attribute_types[i] = schema_attr_type(schema, colID);

            switch (meta->attribute_types[i]) {
            case CONTINUOUS:
                meta->num_discrete_values[i] = 0;
                meta->discrete_attribute_map[i] = NULL;
                break;
            case DISCRETE: {
                // Copy in possible values for the variable.
                uint arity = schema_attr_arity(schema, colID);
                char** values = e_calloc(arity, sizeof(char*));
                uint valID = 0;
                meta->num_discrete_values[i] = arity;
                for (valID = 0; valID < arity; ++valID) {
                    values[valID] = e_strdup(
                        schema_get_discrete_value(schema, colID, valID));
                }
                meta->discrete_attribute_map[i] = values;
                break;
            }
            case UNKNOWN: // fall through
            case CLASS:   // fall through
            case EXCLUDE: // fall through
            default:
                fprintf(stderr, "error: only expected DISCRETE and CONTINUOUS active predictors\n");
                exit(-1);
                break;
            }

            i += 1;
        }
    }
    meta->Missing = NULL;

    if(meta->num_classes)
    {
      for(i = 0; i < meta->num_classes; i++)
        free(meta->class_names[i]);
      free(meta->class_names);
      free(meta->num_examples_per_class);
    }

    // Fill in the meta data for the class variable.
    meta->num_classes = schema_num_classes(schema);
    {
        uint valID = 0;
        uint arity = meta->num_classes;
        // Cosmin added if statement (09/01/2020)
        if (meta->class_names != NULL)
            free(meta->class_names);
        meta->class_names = e_calloc(arity, sizeof(char*));
        for (valID = 0; valID < arity; ++valID) {
            meta->class_names[valID] = e_strdup(schema_get_class_value(schema, valID));
        }
        // Cosmin added if statement (09/01/2020)
        if (meta->num_examples_per_class != NULL)
            free(meta->num_examples_per_class);
        meta->num_examples_per_class = e_calloc(arity, sizeof(int));
    }
    if (class != NULL) {
        memset(class, 0, sizeof(CV_Class));
        uint valID = 0;
        uint arity = meta->num_classes;
        class->num_classes = arity;
        free(class->class_var_name);
        class->class_var_name = e_strdup(schema_class_name(schema));
        // Note: we could alias class->class_names to
        // meta->class_names, but this shouldn't cost much space and
        // will make freeing safely much easier.
        free(class->class_names);
        class->class_names = e_calloc(arity, sizeof(char*));
        for (valID = 0; valID < arity; ++valID) {
            class->class_names[valID] = e_strdup(meta->class_names[valID]);
        }
        class->class_frequencies = e_calloc(arity, sizeof(int));
        class->thresholds = e_calloc(arity, sizeof(float));
    }


    //
    // Update the arguments structure.
    //
    // REVIEW-2012-03-27-ArtM: this is brittle, and should be rewritten to avoid modifying arguments struct
    //

    args->truth_column = schema_class_column(schema) + 1;

    // Update the skipped list if requested.  This is necessary if the
    // .names file had some variables marked 'exclude' that were not 
    // listed in command line options.
    //
    // REVIEW-2012-03-27-ArtM: shouldn't this always be TRUE?
    if (update_skip_list) {
        uint num_exclude = num_columns - meta->num_attributes - 1; // -1 for target
        if (num_exclude == 0) {
            args->num_skipped_features = 0;
            free(args->skipped_features);
            args->skipped_features = NULL;
        }
        else {
            Boolean* old_excluded = e_calloc(num_columns, sizeof(Boolean));
            for (i = 0; i < args->num_skipped_features; ++i) {
                old_excluded[ manual_exclude[i] ] = TRUE;
            }

            args->num_skipped_features = 0;
            free(args->skipped_features);
            args->skipped_features = e_calloc(num_exclude, sizeof(int));

            for (colID = 0; colID < num_columns; ++colID) {
                if (!schema_attr_is_active(schema, colID)) {
                    args->skipped_features[ args->num_skipped_features ] = colID + 1;
                    args->num_skipped_features += 1;

                    // If attribute to exclude was not listed on the
                    // command line, warn the user.
                    if (!old_excluded[colID]) {
                        fprintf(stderr, 
                                "WARNING: Feature in column %d will be excluded as specified in %s\n",
                                colID+1, args->names_file);
                    }
                }
            }

            free(old_excluded);
        }        
    }

    //_show_skipped_features(stdout, "Final skipped feature list: ", args->num_skipped_features, args->skipped_features);
    
    //
    // Release internal schema object.
    //
    free_schema(schema);
    free(manual_exclude);
    
    return 1;
}

void init_predictions(CV_Metadata data, FC_Dataset *dataset, int iter, Args_Opts args) {
    #ifdef HAVE_AVATAR_FCLIB
    int i, j, k;
    int num_steps;
    char *mesh_name;
    FC_Mesh *meshes;
    FC_Sequence seq;
    FC_Variable ***vars; // vars[0][i][0] is for test fold num
                         // vars[1][i][0] is for truth
                         // vars[2][i][0] is for prediction
                         // vars[3][i][0] is for margin
                         // vars[4-?][i][0] are for probabilities
    
    // Create dataset
    av_exitIfErrorPrintf((AV_ReturnCode) fc_createDataset("Predictions", dataset), "Failed to create Predictions dataset\n");
    // Malloc meshes
    meshes = (FC_Mesh *)malloc(data.exo_data.num_seq_meshes * sizeof(FC_Mesh));
    // Create 1-step sequence
    av_exitIfErrorPrintf((AV_ReturnCode) fc_createSequence(*dataset, "TestingTimesteps", &seq), "Failed to create new sequence\n");
    double *seq_coords;
    seq_coords = (double *)malloc(args.num_test_times * sizeof(double));
    for (i = 0; i < args.num_test_times; i++)
        seq_coords[i] = (double)i;
    av_exitIfErrorPrintf((AV_ReturnCode) fc_setSequenceCoords(seq, args.num_test_times, FC_DT_DOUBLE, seq_coords),
                         "Failed to create sequence\n");
    free(seq_coords);
    // Malloc vars
    int num_vars = 3;
    int probs_start_at = num_vars;
    if (args.output_margins == TRUE) {
        num_vars++;
        probs_start_at = num_vars;
    }
    if (args.output_probabilities == TRUE || args.output_laplacean == TRUE)
        num_vars += data.num_classes;
    vars = (FC_Variable ***)malloc(num_vars * sizeof(FC_Variable **));
    for (i = 0; i < num_vars; i++)
        vars[i] = (FC_Variable **)malloc(data.exo_data.num_seq_meshes * sizeof(FC_Variable*));

    for (i = 0; i < data.exo_data.num_seq_meshes; i++) {
        // Get name of mesh
        if ((AV_ReturnCode) fc_getMeshName(data.exo_data.seq_meshes[i], &mesh_name) != AV_SUCCESS) {
            mesh_name = (char *)malloc(1024 * sizeof(char));
            sprintf(mesh_name, "Mesh%06d", i);
        }
        // Create new mesh with same name
        av_exitIfErrorPrintf((AV_ReturnCode) fc_createMesh(*dataset, mesh_name, &meshes[i]),
                             "Failed to create mesh %d for Predictions\n", i);
        free(mesh_name);
        // Set coordinates
        int dim, vert, elems;
        double *coords;
        FC_ElementType elem_type;
        int *conns;
        av_exitIfErrorPrintf((AV_ReturnCode) fc_getMeshDim(data.exo_data.seq_meshes[i], &dim),
                             "Failed to get mesh dim %d for Predictions\n", i);
        av_exitIfErrorPrintf((AV_ReturnCode) fc_getMeshNumVertex(data.exo_data.seq_meshes[i], &vert),
                             "Failed to get vertices %d for Predictions\n", i);
        av_exitIfErrorPrintf((AV_ReturnCode) fc_getMeshCoordsPtr(data.exo_data.seq_meshes[i], &coords),
                             "Failed to get mesh coords %d for Predictions\n", i);
        av_exitIfErrorPrintf((AV_ReturnCode) fc_setMeshCoords(meshes[i], dim, vert, coords),
                             "Failed to set mesh coords %d for Predictions\n", i);

        // Set element connectivities
        av_exitIfErrorPrintf((AV_ReturnCode) fc_getMeshNumElement(data.exo_data.seq_meshes[i], &elems),
                             "Failed to get num elements %d for Predictions\n", i);
        av_exitIfErrorPrintf((AV_ReturnCode) fc_getMeshElementType(data.exo_data.seq_meshes[i], &elem_type),
                             "Failed to get element type %d for Predictions\n", i);
        av_exitIfErrorPrintf((AV_ReturnCode) fc_getMeshElementConnsPtr(data.exo_data.seq_meshes[i], &conns),
                             "Failed to get mesh conns %d for Predictions\n", i);
        av_exitIfErrorPrintf((AV_ReturnCode) fc_setMeshElementConns(meshes[i], elem_type, elems, conns),
                             "Failed to set mesh conns %d for Predictions\n", i);
        
        // Create variables for fold_number, truth, and prediction
        av_exitIfErrorPrintf((AV_ReturnCode) fc_createSeqVariable(meshes[i], seq, "Fold", &num_steps, &vars[0][i]),
                             "Failed to create fold_num var for mesh %d\n", i);
        av_exitIfErrorPrintf((AV_ReturnCode) fc_createSeqVariable(meshes[i], seq, "Truth", &num_steps, &vars[1][i]),
                             "Failed to create truth var for mesh %d\n", i);
        av_exitIfErrorPrintf((AV_ReturnCode) fc_createSeqVariable(meshes[i], seq, "Prediction", &num_steps, &vars[2][i]),
                             "Failed to create prediction var for mesh %d\n", i);
        // Create variables for margins if requested
        if (args.output_margins == TRUE)
            av_exitIfErrorPrintf((AV_ReturnCode) fc_createSeqVariable(meshes[i], seq, "Margin", &num_steps, &vars[3][i]),
                                 "Failed to create margin var for mesh %d\n", i);
        // Create variables for probabilities if requested
        if (args.output_probabilities == TRUE) {
            for (j = 0; j < data.num_classes; j++)
                av_exitIfErrorPrintf((AV_ReturnCode) fc_createSeqVariable(meshes[i], seq, data.class_names[j], &num_steps,
                              &vars[j+probs_start_at][i]), "Failed to create probability var %d for mesh %d\n", j, i);
        }
        
        // Initialize everything to -1
        double *init;
        int num_data_pts = data.global_offset[i+1] - data.global_offset[i];
        init = (double *)malloc(num_data_pts * sizeof(double));
        for (j = 0; j < num_data_pts; j++)
            init[j] = -1.0;
        for (j = 0; j < args.num_test_times; j++) {
            av_exitIfErrorPrintf((AV_ReturnCode) fc_setVariableData(vars[0][i][j], num_data_pts, 1, data.exo_data.assoc_type,
                                                    FC_MT_SCALAR, FC_DT_DOUBLE, init),
                                                    "Failed to initialize fold_num data %d for Predictions\n", i);
            av_exitIfErrorPrintf((AV_ReturnCode) fc_setVariableData(vars[1][i][j], num_data_pts, 1, data.exo_data.assoc_type,
                                                    FC_MT_SCALAR, FC_DT_DOUBLE, init),
                                                    "Failed to initialize truth data %d for Predictions\n", i);
            av_exitIfErrorPrintf((AV_ReturnCode) fc_setVariableData(vars[2][i][j], num_data_pts, 1, data.exo_data.assoc_type,
                                                    FC_MT_SCALAR, FC_DT_DOUBLE, init),
                                                    "Failed to initialize prediction data %d for Predictions\n", i);
            if (args.output_margins == TRUE)
                av_exitIfErrorPrintf((AV_ReturnCode) fc_setVariableData(vars[3][i][j], num_data_pts, 1, data.exo_data.assoc_type,
                                                        FC_MT_SCALAR, FC_DT_DOUBLE, init),
                                                        "Failed to initialize margin data %d for Predictions\n", i);
            if (args.output_probabilities == TRUE) {
                for (k = 0; k < data.num_classes; k++)
                    av_exitIfErrorPrintf((AV_ReturnCode) fc_setVariableData(vars[k+probs_start_at][i][j], num_data_pts, 1,
                                                            data.exo_data.assoc_type, FC_MT_SCALAR, FC_DT_DOUBLE, init),
                                                            "Failed to create probability var %d for mesh %d\n", j, i);
            }
        }
        free(init);

    }
    char *pred_filename;
    if (iter == -1) {
        pred_filename = av_strdup(args.predictions_file);
    } else {
        // Force build_output_filename to handle this like N-fold even if it's not
        Boolean do_X = args.do_5x2_cv;
        Boolean do_N = args.do_nfold_cv;
        args.do_5x2_cv = FALSE;
        args.do_nfold_cv = TRUE;
        pred_filename = build_output_filename(iter, args.predictions_file, args);
        // Revert to original settings
        args.do_5x2_cv = do_X;
        args.do_nfold_cv = do_N;
    }
    av_exitIfErrorPrintf((AV_ReturnCode) fc_writeDataset(*dataset, pred_filename, FC_FT_EXODUS),
                         "Failed to write the predictions exodus file\n");
    free(pred_filename);
    #else
    av_missingFCLIB();
    #endif
}

//Modified by DACIESL June-04-08: Laplacean Estimates
//added consideration for cases of Laplacean Estimate output
int store_predictions_text(CV_Subset test_data, Vote_Cache cache, CV_Matrix matrix, CV_Voting votes, CV_Prob_Matrix prob_matrix, int fold, Args_Opts args) {
    int line, j, k;
    char *p_filename = NULL;
    char *op_filename = NULL;
    char *t_filename = NULL;
    FILE *p_fh = NULL, *op_fh = NULL, *t_fh = NULL;
    //char strbuf[262144];

    char *strbuf;
    int fold_num = fold % args.num_folds + 1;
    int iter_num = fold / args.num_folds + 1;
    static int *pred_data;
    static double **class_data;
    static double **overall_class_data; // For the overall prediction file for 5x2cv
    static int num_malloc_samples = 16;
    
    if (fold <= 0) {
        // First time through so set up pred_data[][] array
        //printf("Initializing pred_data (%d) and class_data (%d)\n", num_malloc_samples, test_data.meta.num_classes);
        pred_data = (int *)malloc(num_malloc_samples * sizeof(int));
        if (args.output_probabilities == TRUE || args.output_margins == TRUE ||
            args.output_laplacean == TRUE || args.do_5x2_cv == TRUE) {
            class_data = (double **)malloc(test_data.meta.num_classes * sizeof(double *));
            for (j = 0; j < test_data.meta.num_classes; j++)
                class_data[j] = (double *)malloc(num_malloc_samples * sizeof(double));
        }
        // This is needed to predict the class so initialize even if probabilities weren't requested
        if (args.do_5x2_cv == TRUE) {
            overall_class_data = (double **)malloc(test_data.meta.num_classes * sizeof(double *));
            for (j = 0; j < test_data.meta.num_classes; j++)
                overall_class_data[j] = (double *)calloc(num_malloc_samples, sizeof(double));
        }
    }
    
    // Run through the samples in this fold and compute values
    for (line = 0; line < test_data.meta.num_examples; line++) {
        CV_Example e = test_data.examples[line];
        // Make sure we have enough space
        while (e.global_id_num+1 > num_malloc_samples) {
            int old_malloc = num_malloc_samples;
            num_malloc_samples *= 2;
            //printf("Reallocating to %d\n", num_malloc_samples);
            pred_data = (int *)realloc(pred_data, num_malloc_samples * sizeof(int));
            if (args.output_probabilities == TRUE || args.output_margins == TRUE ||
                args.output_laplacean == TRUE || args.do_5x2_cv == TRUE) {
                for (j = 0; j < test_data.meta.num_classes; j++)
                    class_data[j] = (double *)realloc(class_data[j], num_malloc_samples * sizeof(double));
            }
            if (args.do_5x2_cv == TRUE) {
                for (j = 0; j < test_data.meta.num_classes; j++) {
                    overall_class_data[j] = (double *)realloc(overall_class_data[j], num_malloc_samples * sizeof(double));
                    // Need to zero-out the new pieces
                    for (k = old_malloc; k < num_malloc_samples; k++)
                        overall_class_data[j][k] = 0.0;
                }
            }
        }
        
        // Make sure this one is first because matrix.num_classes is also > 0 but for the
        // probabilities for ivoting, we MUST use the Vote_Cache
        if (prob_matrix.num_classes > 0) {
            // Store predicted class value
            if (e.predicted_class_num == -1)
              e.predicted_class_num = find_best_class_from_matrix(line, matrix, args, line, 0);
            pred_data[e.global_id_num] = e.predicted_class_num;
            
            // Compute and store probabilities
            if (args.output_laplacean == TRUE || args.do_5x2_cv == TRUE) {
                for (j = 0; j < prob_matrix.num_classes; j++) {
                    class_data[j][e.global_id_num] = (double)prob_matrix.data[line][j];
                    //printf("%f%s",class_data[j][e.global_id_num],(j==prob_matrix.num_classes-1)?"\n":" ");
                    if (args.do_5x2_cv == TRUE)
                        overall_class_data[j][e.global_id_num] += class_data[j][e.global_id_num];
                }
            }
        } else if (cache.num_classes > 0) {
            // Store predicted class value
            if (e.predicted_class_num == -1)
                e.predicted_class_num = cache.best_test_class[line];
            pred_data[e.global_id_num] = e.predicted_class_num;
            
            // Compute and store probabilities
            if (args.output_probabilities == TRUE || args.output_margins == TRUE || args.do_5x2_cv == TRUE) {
                int normalize = 0;
                for (j = 0; j < cache.num_classes; j++)
                    normalize += cache.class_votes_test[line][j];
                for (j = 0; j < cache.num_classes; j++) {
                    //printf("(1)Setting %d.%03d to %d/%d\n", j, e.global_id_num, cache.class_votes_test[line][j], normalize);
                    class_data[j][e.global_id_num] = (double)cache.class_votes_test[line][j] / (double)normalize;
                    if (args.do_5x2_cv == TRUE)
                        overall_class_data[j][e.global_id_num] += class_data[j][e.global_id_num];
                }
            }
        } else if (matrix.num_classes > 0) {
            // Store predicted class value
            if (e.predicted_class_num == -1)
              e.predicted_class_num = find_best_class_from_matrix(line, matrix, args, line, 0);
            pred_data[e.global_id_num] = e.predicted_class_num;
            
            // Store probabilities
            if (args.output_probabilities == TRUE || args.output_margins == TRUE || args.do_5x2_cv == TRUE) {
                if (args.do_boosting == TRUE) {
                    double sum = 0.0;
                    for (j = 0; j < matrix.num_classes; j++)
                        sum += matrix.data[line][j].Real;
                    for (j = 0; j < matrix.num_classes; j++) {
                        //printf("(2)Setting %d.%03d to %g\n", j, e.global_id_num, matrix.data[line][j].Real);
                        class_data[j][e.global_id_num] = (double)matrix.data[line][j].Real / sum;
                        if (args.do_5x2_cv == TRUE)
                            overall_class_data[j][e.global_id_num] += class_data[j][e.global_id_num];
                    }
                } else {
                    int *part_sums;
                    part_sums = (int *)calloc(matrix.num_classes, sizeof(int));
                    for (j = matrix.additional_cols; j < matrix.num_classifiers + matrix.additional_cols; j++)
                        part_sums[matrix.data[line][j].Integer]++;
                    for (j = 0; j < matrix.num_classes; j++) {
                        //printf("(2)Setting %d.%03d to %d/%d\n", j, e.global_id_num, part_sums[j], matrix.num_classifiers);
                        class_data[j][e.global_id_num] = (double)part_sums[j] / (double)matrix.num_classifiers;
                        if (args.do_5x2_cv == TRUE)
                            overall_class_data[j][e.global_id_num] += class_data[j][e.global_id_num];
                    }
                    free(part_sums);
                }
            }
        } else if (votes.num_classes > 0) {
            // Store predicted class value
            if (e.predicted_class_num == -1)
                e.predicted_class_num = votes.best_class[line];
            pred_data[e.global_id_num] = e.predicted_class_num;
            
            // Store probabilities
            if (args.output_probabilities == TRUE || args.output_margins == TRUE || args.do_5x2_cv == TRUE) {
                double normalize = 0;
                for (j = 0; j < votes.num_classes; j++)
                    normalize += votes.data[line][j];
                for (j = 0; j < votes.num_classes; j++) {
                    //printf("(3)Setting %d.%03d to %f/%f\n", j, e.global_id_num, votes.data[line][j], normalize);
                    class_data[j][e.global_id_num] = votes.data[line][j] / normalize;
                    if (args.do_5x2_cv == TRUE)
                        overall_class_data[j][e.global_id_num] += class_data[j][e.global_id_num];
                }
            }
        } else {
            return 0;
        }
    }
    if (fold == -1 || fold_num == args.num_folds) {
        // This is the last fold for this iteration so write prediction file

    // Open .test or .data file for reading and .pred file for writing
    if (fold > -1) {
        if (args.do_5x2_cv) {
            File_Bits bits = explode_filename(args.predictions_file);
            p_filename = (char *)malloc((strlen(bits.dirname) + strlen(bits.basename) +
                                         num_digits(iter_num) + strlen(bits.extension) + 4) * sizeof(char));
            sprintf(p_filename, "%s/%s-%d.%s", bits.dirname, bits.basename, iter_num, bits.extension);
            if (iter_num == 5)
                op_filename = av_strdup(args.predictions_file);
        } else {
            p_filename = av_strdup(args.predictions_file);
        }
    } else {
        p_filename = av_strdup(args.predictions_file);
    }
    if (args.caller == CROSSVALFC_CALLER)
        t_filename = av_strdup(args.datafile);
    else if (args.caller == AVATARDT_CALLER)
        t_filename = av_strdup(args.test_file);
    else
        return 0;
//printf("Reading '%s'\n", t_filename);
    if ((p_fh = fopen(p_filename, "w")) == NULL) {
        fprintf(stderr, "Failed to open .pred file for write: '%s'\n", p_filename);
        return 0;
    }
    if ((t_fh = fopen(t_filename, "r")) == NULL) {
        fprintf(stderr, "Failed to open .%s file for read: '%s'\n",
                args.caller==CROSSVALFC_CALLER?"data":"test", t_filename);
        return 0;
    }
    free(p_filename);
    free(t_filename);
    if (iter_num == 5) {
        if ((op_fh = fopen(op_filename, "w")) == NULL) {
            fprintf(stderr, "Failed to open .pred file for write: '%s'\n", op_filename);
            return 0;
        }
        free(op_filename);
    }
    
    // Read a line from the .test file and echo, without \n, to .pred file
    // Then print test data results
    
    line = 0; // Line number
    Boolean wrote_labels = FALSE;
    
    //while (fscanf(t_fh, "%[^\n]", strbuf) > 0 || fscanf(t_fh, "%[\n]", strbuf) > 0) {
    while (read_line(t_fh, &strbuf) > 0) {
        //if (strbuf[0] == '\n')
        //    continue;
        
        strip_lt_whitespace(strbuf);
        if (*strbuf == '\0')
            continue;
        
        // Retain comment lines
        if (strbuf[0] == '#') {
            fprintf(p_fh, "%s", strbuf);
            if (args.do_5x2_cv == TRUE && iter_num == 5)
                fprintf(op_fh, "%s", strbuf);
            // If this is the #labels line, then add other column headers
            // Find first non-'#' and non-' ' character and see if it's "labels "
            int i = 0;
            while (strbuf[i] == '#' || strbuf[i] == ' ')
                i++;
            if (! strncmp(strbuf + i, "labels ", 7)) {
                wrote_labels = TRUE;
                fprintf(p_fh, ",Pred");
                if (args.do_5x2_cv == TRUE && iter_num == 5)
                    fprintf(op_fh, ",Pred");
                if (args.output_margins == TRUE) {
                    fprintf(p_fh, ",Margin");
                    if (args.do_5x2_cv == TRUE && iter_num == 5)
                        fprintf(op_fh, ",Margin");
                }
                if (args.output_probabilities == TRUE || args.output_laplacean == TRUE) {
                    int num_classes = 0;
                    if (matrix.num_classes > 0)
                        num_classes = matrix.num_classes;
                    else if (votes.num_classes > 0)
                        num_classes = votes.num_classes;
                    for (j = 0; j < num_classes; j++) {
                        fprintf(p_fh, ",Pr %s", test_data.meta.class_names[j]);
                        if (args.do_5x2_cv == TRUE && iter_num == 5)
                            fprintf(op_fh, ",Pr %s", test_data.meta.class_names[j]);
                    }
                }
            }
            fprintf(p_fh, "\n");
            if (args.do_5x2_cv == TRUE && iter_num == 5)
                fprintf(op_fh, "\n");
            continue;
        }
        
        // If we've gotten here without echoing the #labels line then there wasn't one. Make it up.
        if (wrote_labels == FALSE) {
            wrote_labels = TRUE;
            fprintf(p_fh, "#labels ");
            if (args.do_5x2_cv == TRUE && iter_num == 5)
                fprintf(op_fh, "#labels ");
            int consec_count = -1;
            for (j = 0; j < test_data.meta.num_attributes + args.num_skipped_features + 1; j++) {
                // This is a skipped attribute and we don't know the attribute name so write SKIPPED
              if (find_int(j+1, args.num_skipped_features, args.skipped_features)) {
                    fprintf(p_fh, "SKIPPED,");
                    if (args.do_5x2_cv == TRUE && iter_num == 5)
                        fprintf(op_fh, "SKIPPED,");
                    continue;
                }
                // This is the truth column so write Truth
                if (j == args.truth_column-1) {
                    fprintf(p_fh, "Truth,");
                    if (args.do_5x2_cv == TRUE && iter_num == 5)
                        fprintf(op_fh, "Truth,");
                    continue;
                }
                // This is an attribute that we actually used so print its name.
                consec_count++;
                fprintf(p_fh, "%s,", test_data.meta.attribute_names[consec_count]);
                if (args.do_5x2_cv == TRUE && iter_num == 5)
                    fprintf(op_fh, "%s,", test_data.meta.attribute_names[consec_count]);
            }
            fprintf(p_fh, "Pred");
            if (args.do_5x2_cv == TRUE && iter_num == 5)
                fprintf(op_fh, "Pred");
            if (args.output_margins == TRUE) {
                fprintf(p_fh, ",Margin");
                if (args.do_5x2_cv == TRUE && iter_num == 5)
                    fprintf(op_fh, ",Margin");
            }
            if (args.output_probabilities == TRUE || args.output_laplacean == TRUE) {
                int num_classes = 0;
                //DACIESL: changed this guy up a bit, hope it's ok!
                if(prob_matrix.num_classes > 0)
		  num_classes = prob_matrix.num_classes;
                else if (matrix.num_classes > 0)
                    num_classes = matrix.num_classes;
                else if (votes.num_classes > 0)
                    num_classes = votes.num_classes;
                for (j = 0; j < num_classes; j++) {
                    fprintf(p_fh, ",Pr %s", test_data.meta.class_names[j]);
                    if (args.do_5x2_cv == TRUE && iter_num == 5)
                        fprintf(op_fh, ",Pr %s", test_data.meta.class_names[j]);
                }
            }
            fprintf(p_fh, "\n");
            if (args.do_5x2_cv == TRUE && iter_num == 5)
                fprintf(op_fh, "\n");
        }
        // Duplicate .test or .data file line
        fprintf(p_fh, "%s", strbuf);
        if (args.do_5x2_cv == TRUE && iter_num == 5)
            fprintf(op_fh, "%s", strbuf);
        free(strbuf);
        //CV_Example e = test_data.examples[line];
        // Write predicted class value
        //if (e.predicted_class_num == -1)
        //    e.predicted_class_num = find_best_class_from_matrix(line, matrix, args);
        //fprintf(p_fh, ",%s", test_data.meta.class_names[e.predicted_class_num]);
        fprintf(p_fh, ",%s", test_data.meta.class_names[pred_data[line]]);
        
        int num_classes = 0;
        //DACIESL: and again here
        if(prob_matrix.num_classes > 0)
            num_classes = prob_matrix.num_classes;
        else if (matrix.num_classes > 0)
            num_classes = matrix.num_classes;
        else if (votes.num_classes > 0)
            num_classes = votes.num_classes;
        
        if (args.do_5x2_cv == TRUE && iter_num == 5) {
            float probs[num_classes];
            int classes[num_classes];
            for (j = 0; j < num_classes; j++) {
                classes[j] = j;
                //printf("Setting probs[%d] for sample %d to %g\n", j, line, overall_class_data[j][line]/5.0);
                probs[j] = overall_class_data[j][line]/5.0;
            }
            float_int_array_sort(num_classes, probs-1, classes-1);
            fprintf(op_fh, ",%s", test_data.meta.class_names[classes[0]]);
        }
        
        // Write probabilities
        if (args.output_probabilities == TRUE || args.output_margins == TRUE || args.output_laplacean == TRUE) {
        //    int *part_sums;
        //    part_sums = (int *)calloc(matrix.num_classes, sizeof(int));
        //    for (j = matrix.additional_cols; j < matrix.num_classifiers + matrix.additional_cols; j++)
        //        part_sums[matrix.data[line][j]]++;
            if (args.output_margins == TRUE) {
                float probs[num_classes];
                for (j = 0; j < num_classes; j++) {
                    probs[j] = class_data[j][line];
                }
                float_array_sort(num_classes, probs-1);
                fprintf(p_fh, ",%g", probs[num_classes-1] - probs[num_classes-2]);
                if (args.do_5x2_cv == TRUE && iter_num == 5) {
                    for (j = 0; j < num_classes; j++) {
                        probs[j] = overall_class_data[j][line]/5.0;
                    }
                    float_array_sort(num_classes, probs-1);
                    fprintf(op_fh, ",%g", probs[num_classes-1] - probs[num_classes-2]);
                }
            }
            if (args.output_probabilities == TRUE || args.output_laplacean == TRUE) {
                for (j = 0; j < num_classes; j++) {
                    //printf("Writing %d.%03d\n", j, line);
                    fprintf(p_fh, ",%g", class_data[j][line]);
                    if (args.do_5x2_cv == TRUE && iter_num == 5)
                        fprintf(op_fh, ",%g", overall_class_data[j][line]/5.0);
                }
            }
        //        fprintf(p_fh, ",%g", (double)part_sums[j] / (double)matrix.num_classifiers);
        //    free(part_sums);
        }
        // End the line
        fprintf(p_fh, "\n");
        if (args.do_5x2_cv == TRUE && iter_num == 5)
            fprintf(op_fh, "\n");

        line++;
        //if (line > test_data.meta.num_examples) {
        //    fprintf(stderr, "Error: .test file has more examples than expected\n");
        //    exit(-1);
        //}

    }
    fclose(p_fh);
    fclose(t_fh);
    if (args.do_5x2_cv == TRUE && iter_num == 5)
        fclose(op_fh);
    }
    return 1;
}

/*
 * Not used at the current time
 *
int _store_predictions_text(CV_Subset test_data, CV_Matrix matrix, Args_Opts args) {
    int i;
    FILE *fh_r, *fh_w;
    char strbuf[262144], crlf[2];
    int num_elements;
    char **elements;
    int ex_num = -1;
    
    // Open .test file for reading and .pred file for writing
    if ((fh_r = fopen(args.test_file, "r")) == NULL) {
        fprintf(stderr, "Failed to open test file for read: '%s'\n", args.test_file);
        return 0;
    }
    if ((fh_w = fopen(args.predictions_file, "w")) == NULL) {
        fprintf(stderr, "Failed to open predictions file for write: '%s'\n", args.predictions_file);
        return 0;
    }
    
    while (fscanf(fh_r, "%[^\n]", strbuf) > 0) {
        //parse_comma_sep_string(strbuf, &num_elements, &elements);
        parse_delimited_string(',', strbuf, &num_elements, &elements);
        //printf("Processing '%s'\n", strbuf);

        // Just echo comments to output
        if (elements[0][0] == '#') {
            fprintf(fh_w, "%s\n", strbuf);
            fscanf(fh_r, "%[\n]", crlf);
            continue;
        }
        
        // Make sure we have enough elements for all attributes and the true class
        // If not, skip and do not print to pred file
        if (num_elements < test_data.meta.num_attributes+1) {
            fprintf(stderr, "Expected at least %d values but read %d ... skipping\n",
                                                                            test_data.meta.num_attributes+1, num_elements);
            fscanf(fh_r, "%[\n]", crlf);
            continue;
        }
        
        ex_num++;
        
        fprintf(fh_w, "%s", strbuf);
        if (test_data.examples[ex_num].predicted_class_num == -1)
            fprintf(fh_w, ",%s", test_data.meta.class_names[find_best_class_from_matrix(ex_num, matrix, args)]);
        else
            fprintf(fh_w, ",%s", test_data.meta.class_names[test_data.examples[ex_num].predicted_class_num]);
        
        if (args.output_probabilities == TRUE) {
            int *part_sums;
            part_sums = (int *)calloc(matrix.num_classes, sizeof(int));
            for (i = matrix.additional_cols; i < matrix.num_classifiers + matrix.additional_cols; i++)
                part_sums[matrix.data[ex_num][i]]++;
            for (i = 0; i < matrix.num_classes; i++)
                fprintf(fh_w, ",%g", (double)part_sums[i] / (double)matrix.num_classifiers);
            free(part_sums);
        }
        
        fprintf(fh_w, "\n");
        fscanf(fh_r, "%[\n]", crlf);
        
        for (i = 0; i < num_elements; i++)
            free(elements[i]);
        free(elements);
    }
    
    fclose(fh_r);
    fclose(fh_w);
    
    return 1;
}
 *
 */

int store_predictions(CV_Subset test_data, Vote_Cache cache, CV_Matrix matrix, CV_Voting votes, CV_Prob_Matrix prob_matrix, FC_Dataset dataset, int fold, Args_Opts args) {
    #ifdef HAVE_AVATAR_FCLIB
    int i, j, k, m;
    FC_Variable ***vars; // vars[0][i][j] is for test fold num
                         // vars[1][i][j] is for truth
                         // vars[2][i][j] is for prediction
                         // vars[3][i][j] is for margins
                         // vars[4-?][i][j] are for probabilities
    FC_Mesh *meshes;
    int num_meshes;
    double **data;
    double ***class_data;
    int iter_num = fold / args.num_folds;// + 1;
    static double **overall_class_data; // For the overall prediction file for 5x2cv
    
    if (fold <= 0 && args.do_5x2_cv == TRUE) {
        // This is needed to predict the class so initialize even if probabilities weren't requested
        overall_class_data = (double **)malloc(test_data.meta.num_classes * sizeof(double *));
        for (j = 0; j < test_data.meta.num_classes; j++)
            overall_class_data[j] = (double *)calloc(test_data.meta.num_examples, sizeof(double));
    }
    
    // Get the meshes from the dataset
    av_exitIfErrorPrintf((AV_ReturnCode) fc_getMeshes(dataset, &num_meshes, &meshes), "Failed to get meshes from Predictions dataset");
    if (num_meshes != test_data.meta.exo_data.num_seq_meshes) {
        fprintf(stderr, "Expected %d but got %d meshes from Predictions dataset\n",
                        test_data.meta.exo_data.num_seq_meshes, num_meshes);
        return 0;
    }
    
    // Malloc vars
    int num_vars = 3;
    int probs_start_at = num_vars;
    if (args.output_margins == TRUE) {
        num_vars++;
        probs_start_at = num_vars;
    }
    if (args.output_probabilities == TRUE || args.output_laplacean == TRUE)
        num_vars += test_data.meta.num_classes;
    vars = (FC_Variable ***)malloc(num_vars * sizeof(FC_Variable **));
    for (i = 0; i < num_vars; i++)
        vars[i] = (FC_Variable **)malloc(test_data.meta.exo_data.num_seq_meshes * sizeof(FC_Variable*));
    
    for (i = 0; i < test_data.meta.exo_data.num_seq_meshes; i++) {
        int num_data_pts = test_data.meta.global_offset[i+1]-test_data.meta.global_offset[i];
        //int global_id_base = fclib2global(i, 0, test_data.num_fclib_seq, test_data.meta.global_offset);
        //int mesh = i % test_data.meta.exo_data.num_seq_meshes;
        //int timestep = i / test_data.meta.exo_data.num_seq_meshes;
        
        data = (double **)malloc(args.num_test_times * sizeof(double *));        
        for (j = 0; j < args.num_test_times; j++)
            data[j] = (double *)malloc(num_data_pts * sizeof(double));
        
        // Update fold_num var
        av_exitIfErrorPrintf((AV_ReturnCode) _copy_and_delete_orig(meshes[i], "Fold", &vars[0][i], &data),
                             "Error copying fold variable");
        for (j = 0; j < test_data.meta.num_examples; j++) {
            if (test_data.examples[j].fclib_seq_num % test_data.meta.exo_data.num_seq_meshes == i) {
                // This fclib sequence corresponds to the exodus mesh we are working on
                
                // Get the timestep that this fclib sequence corresponds to
                int timestep = test_data.examples[j].fclib_seq_num / test_data.meta.exo_data.num_seq_meshes;
                data[timestep][test_data.examples[j].fclib_id_num] = test_data.examples[j].containing_fold_num;
                //printf("Setting %d\n", test_data.examples[j].global_id_num);
            }
        }
        for (j = 0; j < args.num_test_times; j++)
            av_exitIfErrorPrintf((AV_ReturnCode) fc_setVariableData(vars[0][i][j], num_data_pts, 1, test_data.meta.exo_data.assoc_type,
                                                    FC_MT_SCALAR, FC_DT_DOUBLE, (void *)data[j]),
                                                "Failed to set pred fold_num data %d for timestep on mesh %d\n", j, i);

        // Update truth var
        av_exitIfErrorPrintf((AV_ReturnCode) _copy_and_delete_orig(meshes[i], "Truth", &vars[1][i], &data),
                             "Error copying truth variable");
        for (j = 0; j < test_data.meta.num_examples; j++) {
            if (test_data.examples[j].fclib_seq_num % test_data.meta.exo_data.num_seq_meshes == i) {
                // This fclib sequence corresponds to the exodus mesh we are working on
                
                // Get the timestep that this fclib sequence corresponds to
                int timestep = test_data.examples[j].fclib_seq_num / test_data.meta.exo_data.num_seq_meshes;
                data[timestep][test_data.examples[j].fclib_id_num] = test_data.examples[j].containing_class_num;
            }
        }
        for (j = 0; j < args.num_test_times; j++)
            av_exitIfErrorPrintf((AV_ReturnCode) fc_setVariableData(vars[1][i][j], num_data_pts, 1, test_data.meta.exo_data.assoc_type,
                                                    FC_MT_SCALAR, FC_DT_DOUBLE, (void *)data[j]),
                                                "Failed to set pred truth data %d for timestep on mesh %d\n", j, i);

        // Update prediction var
        av_exitIfErrorPrintf((AV_ReturnCode) _copy_and_delete_orig(meshes[i], "Prediction", &vars[2][i], &data),
                             "Error copying prediction variable");
        for (j = 0; j < test_data.meta.num_examples; j++) {
            if (test_data.examples[j].fclib_seq_num % test_data.meta.exo_data.num_seq_meshes == i) {
                if (test_data.examples[j].predicted_class_num < 0) {
                    if (matrix.num_classes > 0)
                        test_data.examples[j].predicted_class_num = find_best_class_from_matrix(j, matrix, args, j, 0);
                    else if (votes.num_classes > 0)
                        test_data.examples[j].predicted_class_num = votes.best_class[j];
                }
                // Get the timestep that this fclib sequence corresponds to
                int timestep = test_data.examples[j].fclib_seq_num / test_data.meta.exo_data.num_seq_meshes;
                data[timestep][test_data.examples[j].fclib_id_num] = test_data.examples[j].predicted_class_num;
            }
        }
        for (j = 0; j < args.num_test_times; j++)
            av_exitIfErrorPrintf((AV_ReturnCode) fc_setVariableData(vars[2][i][j], num_data_pts, 1, test_data.meta.exo_data.assoc_type,
                                                    FC_MT_SCALAR, FC_DT_DOUBLE, (void *)data[j]),
                                            "Failed to set pred prediction data %d for timestep on mesh %d\n", j, i);

        // Update probability vars
        if (args.output_probabilities == TRUE || args.output_laplacean == TRUE || args.output_margins == TRUE) {
            
            int num_classes = 0;
            if (prob_matrix.num_classes > 0)
	        num_classes = prob_matrix.num_classes;
            else if (matrix.num_classes > 0)
                num_classes = matrix.num_classes;
            else if (votes.num_classes > 0)
                num_classes = votes.num_classes;
            class_data = (double ***)malloc(num_classes * sizeof(double **));
            for (j = 0; j < num_classes; j++) {
                class_data[j] = (double **)malloc(args.num_test_times * sizeof(double *));
                for (k = 0; k < args.num_test_times; k++) 
                    class_data[j][k] = (double *)malloc(num_data_pts * sizeof(double));
            }
            
            for (j = 0; j < test_data.meta.num_classes; j++)
                av_exitIfErrorPrintf((AV_ReturnCode) _copy_and_delete_orig(meshes[i], test_data.meta.class_names[j],
                                                           &vars[probs_start_at+j][i], &class_data[j]),
                                     "Error copying '%s' variable", test_data.meta.class_names[j]);
            int *part_sums;
            for (j = 0; j < test_data.meta.num_examples; j++) {
                if (test_data.examples[j].fclib_seq_num % test_data.meta.exo_data.num_seq_meshes == i) {
                    // Get the timestep that this fclib sequence corresponds to
                    int timestep = test_data.examples[j].fclib_seq_num / test_data.meta.exo_data.num_seq_meshes;

                    //Edited by DACIESL to support laplacean probability outputs
                    if (prob_matrix.num_classes > 0) {
                        for (m = 0; m < prob_matrix.num_classes; m++) {
	                    class_data[m][timestep][test_data.examples[j].fclib_id_num] = (double)prob_matrix.data[j][m];
                            if (args.do_5x2_cv == TRUE)
                                overall_class_data[m][test_data.examples[j].fclib_id_num] +=
                                                          class_data[m][timestep][test_data.examples[j].fclib_id_num];
                        }
                    } else if (cache.num_classes > 0) {
                        // Compute and store probabilities
                        int normalize = 0;
                        for (m = 0; m < cache.num_classes; m++)
                            normalize += cache.class_votes_test[j][m];
                        for (m = 0; m < cache.num_classes; m++) {
                            //printf("Setting %d.%03d\n", j, e.global_id_num);
                            class_data[m][timestep][test_data.examples[j].fclib_id_num] =
                                                             (double)cache.class_votes_test[j][m] / (double)normalize;
                            if (args.do_5x2_cv == TRUE)
                                overall_class_data[m][test_data.examples[j].fclib_id_num] +=
                                                          class_data[m][timestep][test_data.examples[j].fclib_id_num];
                        }
                    } else if (matrix.num_classes > 0) {
                        if (args.do_boosting == TRUE) {
                            for (m = 0; m < matrix.num_classes; m++) {
                                class_data[m][timestep][test_data.examples[j].fclib_id_num] = (double)matrix.data[j][m].Real;
                                if (args.do_5x2_cv == TRUE)
                                    overall_class_data[m][test_data.examples[j].fclib_id_num] +=
                                                          class_data[m][timestep][test_data.examples[j].fclib_id_num];
                            }
                        } else {
                            part_sums = (int *)calloc(matrix.num_classes, sizeof(int));
                            for (m = matrix.additional_cols; m < matrix.num_classifiers + matrix.additional_cols; m++)
                                part_sums[matrix.data[j][m].Integer]++;
                            for (m = 0; m < matrix.num_classes; m++) {
                                class_data[m][timestep][test_data.examples[j].fclib_id_num] =
                                                                (double)part_sums[m] / (double)matrix.num_classifiers;
                                if (args.do_5x2_cv == TRUE) {
                                    overall_class_data[m][test_data.examples[j].fclib_id_num] +=
                                                          class_data[m][timestep][test_data.examples[j].fclib_id_num];
                                }
                            }
                            free(part_sums);
                        }
                    } else if (votes.num_classes > 0) {
                        double normalize = 0.0;
                        for (m = 0; m < votes.num_classes; m++)
                            normalize += votes.data[j][m];
                        for (m = 0; m < votes.num_classes; m++) {
                            class_data[m][timestep][test_data.examples[j].fclib_id_num] = votes.data[j][m] / normalize;
                            if (args.do_5x2_cv == TRUE)
                                overall_class_data[m][test_data.examples[j].fclib_id_num] +=
                                                          class_data[m][timestep][test_data.examples[j].fclib_id_num];
                        }
                    }
                }
            }
            if (args.output_margins == TRUE) {
                av_exitIfErrorPrintf((AV_ReturnCode) _copy_and_delete_orig(meshes[i], "Margin", &vars[3][i], &data),
                                     "Error copying margin variable");
                for (j = 0; j < test_data.meta.num_examples; j++) {
                    if (test_data.examples[j].fclib_seq_num % test_data.meta.exo_data.num_seq_meshes == i) {
                        int timestep = test_data.examples[j].fclib_seq_num / test_data.meta.exo_data.num_seq_meshes;
                        float probs[test_data.meta.num_classes];
                        for (m = 0; m < test_data.meta.num_classes; m++)
                            probs[m] = class_data[m][timestep][test_data.examples[j].fclib_id_num];
                        float_array_sort(test_data.meta.num_classes, probs-1);
                        data[timestep][test_data.examples[j].fclib_id_num] = probs[num_classes-1] - probs[num_classes-2];
                    }
                }
                for (j = 0; j < args.num_test_times; j++)
                    av_exitIfErrorPrintf((AV_ReturnCode) fc_setVariableData(vars[3][i][j], num_data_pts, 1, test_data.meta.exo_data.assoc_type,
                                         FC_MT_SCALAR, FC_DT_DOUBLE, (void *)data[j]),
                                        "Failed to set pred margin data %d for timestep on mesh %d\n", j, i);
            }
            
            if (args.output_probabilities == TRUE || args.output_laplacean == TRUE)
                for (j = 0; j < test_data.meta.num_classes; j++)
                    for (k = 0; k < args.num_test_times; k++)
                        av_exitIfErrorPrintf((AV_ReturnCode) fc_setVariableData(vars[probs_start_at+j][i][k], num_data_pts, 1,
                                                                test_data.meta.exo_data.assoc_type, FC_MT_SCALAR,
                                                                FC_DT_DOUBLE, (void *)(class_data[j][k])),
                                 "Failed to set pred class %d probability data for timestep %d on mesh %d\n", j, k, i);
            
            for (j = 0; j < matrix.num_classes; j++) {
                for (k = 0; k < args.num_test_times; k++) 
                    free(class_data[j][k]);
                free(class_data[j]);
            }
            free(class_data);
        }
        
        for (j = 0; j < args.num_test_times; j++)
            free(data[j]);
        free(data);
    }
    
    char *pred_filename;
    if (args.do_5x2_cv == TRUE) {
        // Force build_output_filename to handle this like N-fold even if it's not
        args.do_5x2_cv = FALSE;
        args.do_nfold_cv = TRUE;
        pred_filename = build_output_filename(iter_num, args.predictions_file, args);
        // Revert to original settings
        args.do_5x2_cv = TRUE;
        args.do_nfold_cv = FALSE;
    } else {
        pred_filename = av_strdup(args.predictions_file);
    }
    av_exitIfErrorPrintf((AV_ReturnCode) fc_rewriteDataset(dataset, pred_filename, FC_FT_EXODUS),
                         "Failed to write the predictions exodus file: %s\n", pred_filename);
    free(pred_filename);

    return 1;
    #else
    av_missingFCLIB();
    return 0;
    #endif
}

AV_ReturnCode _copy_and_delete_orig(FC_Mesh mesh, char *var_name, FC_Variable **new_var, double ***data) {
    #ifdef HAVE_AVATAR_FCLIB
    int i, j;
    AV_ReturnCode rc;
    int num_steps;
    FC_Variable *old_var;
    int num_data_pts, num_comps;
    FC_AssociationType assoc_type;
    FC_MathType math_type;
    FC_DataType data_type;
    FC_Sequence seq;
    
    rc = fc_getOrGenerateUniqueSeqVariableByName(mesh, var_name, &num_steps, &old_var);
    if (rc != AV_SUCCESS) {
        av_printfErrorMessage("Failed to get '%s' variable", var_name);
        return rc;
    }
    (*new_var) = (FC_Variable *)malloc(num_steps * sizeof(FC_Variable));

    rc = fc_getVariableInfo(old_var[0], &num_data_pts, &num_comps, &assoc_type, &math_type, &data_type);
    if (rc != AV_SUCCESS) {
        fc_printfErrorMessage("Failed to get variable info for '%s'", var_name);
        return rc;
    }
    
    // Should have only gotten a single variable so assume it
    rc = fc_getSequenceFromSeqVariable(num_steps, old_var, &seq);
    if (rc != AV_SUCCESS) {
        fc_printfErrorMessage("Failed to get sequence for '%s'", var_name);
        return rc;
    }
    int new_num_steps;
    rc = fc_createSeqVariable(mesh, seq, var_name, &new_num_steps, new_var);
    if (rc != AV_SUCCESS) {
        fc_printfErrorMessage("Failed to create new '%s' variable", var_name);
        return rc;
    }
    
    for (i = 0; i < num_steps; i++) {
        void *data_ptr;
        rc = fc_getVariableDataPtr(old_var[i], &data_ptr);
        if (rc != AV_SUCCESS) {
            fc_printfErrorMessage("Failed to get big data for '%s' variable at step %d", var_name, i);
            return rc;
        }
        for (j = 0; j < num_data_pts; j++) {
            //printf("Looking for data point %d/%d\n", i, j);
            double d = *((double *)data_ptr + j);
            //printf("Found data point %d/%d: %f\n", i, j, d);
            (*data)[i][j] = d;
            (*data)[i][j] = *((double *)data_ptr + j);
        }
    }
    rc = fc_deleteSeqVariable(num_steps, old_var);
    if (rc != AV_SUCCESS) {
        fc_printfErrorMessage("Failed to delete old '%s' variable", var_name);
        return rc;
    }

    return AV_SUCCESS;
    #else
    av_missingFCLIB();
    return AV_ERROR;
    #endif
}

static BST_Node **tree;
static Tree_Bookkeeping *books;
static int *high;
static int *low;

void init_att_handling(CV_Metadata meta) {
    int i;
    tree = (BST_Node **)malloc(meta.num_attributes * sizeof(BST_Node *));
    books = (Tree_Bookkeeping *)malloc(meta.num_attributes * sizeof(Tree_Bookkeeping));
    low = (int *)malloc(meta.num_attributes * sizeof(int));
    high = (int *)malloc(meta.num_attributes * sizeof(int));
    
    // Initialize the tree for each attribute on the first example only
    // Even though we're only using the tree for continuous attributes.
    // MAY WANT TO LOOK INTO THIS FOR OPTIMIZATION LATER
    for (i = 0; i < meta.num_attributes; i++) {
        books[i].num_malloced_nodes = 1000;
        books[i].next_unused_node = 1;
        books[i].current_node = 0;
        tree[i] = (BST_Node *)malloc(books[i].num_malloced_nodes * sizeof(BST_Node));
    }
}

int process_attribute_char_value(char *att_value, int att_index, CV_Metadata meta) {
    int dv = -1;
    
    // If the attribute value is "?" this is a missing value so skip it. We'll fill it in later
    if (meta.attribute_types[att_index] == DISCRETE) {
        if (strcmp(att_value, "?")) {
            // Add discrete attribute value to array for missing value computation
            dv = translate_discrete(meta.discrete_attribute_map[att_index], meta.num_discrete_values[att_index], att_value);
            if (dv == -1) {
                /*
                printf("For attribute %d failed to index %s into [", att_index+1, att_value);
                for (i = 0; i < meta.num_discrete_values[att_index]; i++)
                    printf("%s, ", meta.discrete_attribute_map[att_index][i]);
                printf("\b\b]\n");
                */
                // This will definitely cause problems later on when writing the tree file metadata so
                // might as well exit right now
                return 0;
            }
        }
    } else if (meta.attribute_types[att_index] == CONTINUOUS) {
        if (strcmp(att_value, "?"))
            dv = process_attribute_float_value(atof(att_value), att_index);
    }
    
    return dv;
}

int process_attribute_float_value(float att_value, int att_index) {
    int dv = -1;
    //printf("  Handling attribute %d with value %g\n", att_index, att_value);
    if (books[att_index].current_node == 0) {
        tree[att_index][0].value = att_value;
        tree[att_index][0].left = -1;
        tree[att_index][0].right = -1;
        low[att_index] = 0;
        high[att_index] = 0;
        
        // current_node == 0 only triggers the initialization. Otherwise we don't care about it.
        // Increment it only so the initialization is not repeated
        books[att_index].current_node++;
    } else {
        high[att_index] += tree_insert(&tree[att_index], &books[att_index], att_value);
    }
    
    return dv;
}

void create_float_data(CV_Subset *data) {
    int i;
    free(data->float_data);
    free(data->low);
    free(data->high);
    free(data->discrete_used);
    data->float_data = (float **)calloc(data->meta.num_attributes, sizeof(float *));
    data->low = (int *)malloc(data->meta.num_attributes * sizeof(int));
    data->high = (int *)malloc(data->meta.num_attributes * sizeof(int));
    data->discrete_used = (Boolean *)malloc(data->meta.num_attributes * sizeof(Boolean));
    for (i = 0; i < data->meta.num_attributes; i++) {
        data->discrete_used[i] = FALSE;
        data->low[i] = low[i];
        data->high[i] = high[i];
        if (data->meta.attribute_types[i] == CONTINUOUS) {
            data->float_data[i] = (float *)malloc((data->high[i] + 1) * sizeof(float));
            // Check that initialization was done which means the tree has some nodes in it.
            // If there is a single testing sample and this attribute is continuous and contains '?'
            //    then the tree was never initialized and tree_to_array with segfault.
            if (books[i].current_node > 0)
                tree_to_array(data->float_data[i], tree[i]);
            //int j;
            //for (j = data->low[i]; j <= data->high[i]; j++)
            //    printf("FLOAT_DATA[%d][%d] = %g\n", i, j, data->float_data[i][j]);
        }
        free(tree[i]);
    }
    free(tree);
    free(books);
    free(high);
    free(low);
}

int add_attribute_char_value(char *a_val, CV_Subset *sub, int e_num, int a_num, int all_a, int line, char *file, int truth_col, char **elements) {
    int num_errors = 0;
    int i;
    
    if (sub->meta.attribute_types[a_num] == DISCRETE) {
        if (! strcmp(a_val, "?")) {
            // Use missing value for this attribute
            sub->examples[e_num].distinct_attribute_values[a_num] = sub->meta.Missing[a_num].Discrete;
        } else {
            //printf("Looking for '%s' in discrete map for attribute %d\n",
            //          elements[all_atts + (a_num < args.truth_column-1 ? 0 : 1)], a_num);
            int dv = translate_discrete(sub->meta.discrete_attribute_map[a_num], sub->meta.num_discrete_values[a_num], a_val);
            if (dv < 0) {
                num_errors++;
                fprintf(stderr, "Invalid value (%s) for attribute %d: line %d of %s\n",
                                 a_val, all_a+1, line, file);
                printf("Failed to find '%s' in [", elements[all_a + (a_num < truth_col-1 ? 0 : 1)]);
                for (i = 0; i < sub->meta.num_discrete_values[a_num]; i++)
                    printf("'%s',", sub->meta.discrete_attribute_map[a_num][i]);
                printf("\b]\n");
            } else {
                sub->examples[e_num].distinct_attribute_values[a_num] = dv;
            }
        }
    } else if (sub->meta.attribute_types[a_num] == CONTINUOUS) {
        if (! strcmp(a_val, "?")) {
            // Use missing value for this attribute
            sub->examples[e_num].distinct_attribute_values[a_num] = translate(sub->float_data[a_num],
                                                                              sub->meta.Missing[a_num].Continuous,
                                                                              0, sub->high[a_num] + 1);
        } else {
            add_attribute_float_value(atof(a_val), sub, e_num, a_num);
        }
    }
    return num_errors;
}

void add_attribute_float_value(float a_val, CV_Subset *sub, int e_num, int a_num) {
    sub->examples[e_num].distinct_attribute_values[a_num] = translate(sub->float_data[a_num], a_val,
                                                                      0, sub->high[a_num] + 1);
}

void datafile_to_string_array(char *file, int *num_lines, char ***data_lines, int *num_comments, char ***leading_comments) {
    int i;
    FILE *fh;
    if ((fh = fopen(file, "r")) == NULL) {
        fprintf(stderr, "ERROR: %s could not be read\n", file);
        exit(-1);
    }
    *num_lines = 0;
    
    // Initialize data_lines and leading_comments
    *data_lines = (char **)malloc(sizeof(char *));
    
    while (read_line(fh, &(*data_lines)[*num_lines]) > 0) {
        (*num_lines)++;
        *data_lines = (char **)realloc(*data_lines, ((*num_lines)+1)*sizeof(char *));
    }
    
    // Count leading comment lines
    *num_comments = 0;
    Boolean in_leading_comments = TRUE;
    for (i = 0; i < *num_lines; i++) {
        if (in_leading_comments == TRUE && strchr((*data_lines)[i], '#') != NULL)
            (*num_comments)++;
        else
            in_leading_comments = FALSE;
    }
    
    // Initialize leading_comments
    (*leading_comments) = (char **)malloc((*num_comments) * sizeof(char *));
    // Copy leading comments from data_lines to leading_comments
    for (i = 0; i < *num_comments; i++)
        (*leading_comments)[i] = av_strdup((*data_lines)[i]);
    
    // Now, go through and remove all comments from data_lines
    int comment_count = 0;
    int orig_num_lines = *num_lines;
    for (i = 0; i < orig_num_lines; i++) {
        if (strchr((*data_lines)[i], '#') != NULL) {
            comment_count++;
            // Update number of data lines
            (*num_lines)--;
        } else {
            // This is a data line so if there have been some comments, copy this line to the right spot
            if (comment_count > 0) {
                free((*data_lines)[i-comment_count]);
                (*data_lines)[i-comment_count] = av_strdup((*data_lines)[i]);
            }
        }
    }
    
    // Shrink down data_lines to remove extraneous lines
    (*data_lines) = (char **)realloc(*data_lines, (*num_lines)*sizeof(char *));
}


/************************************************************
 * MODULE HELPER FUNCTIONS
 ************************************************************/


/*
    This needs to be called after we know how many attributes there are and before we actually read the data
 */
void update_skipped_features(int num_attributes, Args_Opts *args) {
    int i;
/*
   This takes forever when the last included feature is near the beginning and there are a lot of features
 
    // Exclude features above largest id explicity included
    // Do this brute-force-ishly because some feature ids above exclude_all_features_above could already have been
    // explicitly excluded and I don't want to duplicate them in skipped_features
    
    // The currently excluded features are already accounted for in num_attributes so need to
    // exclude up to and including a 1-based id of (num_attributes + args->num_skipped_features)
    if (args->exclude_all_features_above > 0) {
        int total_num_attributes = num_attributes + args->num_skipped_features;
        for (i = args->exclude_all_features_above+1; i <= total_num_attributes; i++) {
            if (! find_int(i, args->num_skipped_features, args->skipped_features)) {
                // is not already on the list so add it to the list
                args->num_skipped_features++;
                if (args->num_skipped_features == 1)
                    args->skipped_features = (int *)malloc(args->num_skipped_features * sizeof(int));
                else
                    args->skipped_features =
                                    (int *)realloc(args->skipped_features, args->num_skipped_features * sizeof(int));
                args->skipped_features[args->num_skipped_features-1] = i;
            }
        }
        printf("Currently excluding %d features:", args->num_skipped_features);
        for (i = 0; i < args->num_skipped_features; i++)
            printf(" %d", args->skipped_features[i]);
        printf("\n");
    }
 */
    
    // Exclude features above largest id explicity included
    // Add everything and then go through and remove duplicates later
    
    // The currently excluded features are already accounted for in num_attributes so need to
    // exclude up to and including a 1-based id of (num_attributes + args->num_skipped_features)
    if (args->exclude_all_features_above > 0) {
        //int total_num_attributes = num_attributes + args->num_skipped_features;
        for (i = args->exclude_all_features_above+1; i <= num_attributes; i++) {
            if (i != args->truth_column) {
                args->num_skipped_features++;
                if (args->num_skipped_features == 1)
                {
                    free(args->skipped_features);
                    args->skipped_features = (int *)malloc(args->num_skipped_features * sizeof(int));
                }
                else
                    args->skipped_features =
                                    (int *)realloc(args->skipped_features, args->num_skipped_features * sizeof(int));
                //printf("Adding %d to skip list\n", i);
                args->skipped_features[args->num_skipped_features-1] = i;
            }
        }
        //_show_skipped_features(stdout, "Currently excluding %d features:", args->num_skipped_features, args->skipped_features);
    }
    
}


void 
_show_skipped_features(FILE* fh, const char* msg, int num_skipped, const int* skipped)
{
    fprintf(fh, "%s\n", msg);
    if (num_skipped > 0) {
        char* range = NULL;
        array_to_range(skipped, num_skipped, &range);
        fprintf(fh, "%d Skipped Feature Numbers: %s\n", num_skipped, range);
        free(range);
    }
}
