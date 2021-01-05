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
#include <assert.h>
#include "crossval.h"
#include "options.h"
#include "rw_data.h"
#include "util.h"
#include "memory.h"
#include "distinct_values.h"
#include "tree.h"
#include "array.h"
#include "gain.h"
#include "smote.h"
#include "balanced_learning.h"
#include "skew.h"
#include "version_info.h"
#include "attr_stats.h"
#include "evaluate.h"
#include "avatar_api.h"


struct Avatar_struct{
  CV_Dataset Train_Dataset, Test_Dataset;
  CV_Subset Train_Subset, Test_Subset;
  Vote_Cache Cache;
  AV_SortedBlobArray Train_Sorted_Examples, Test_Sorted_Examples;
  FC_Dataset ds, pred_prob;
  CV_Partition Partitions;
  DT_Ensemble* Test_Ensembles;
  CV_Class Class;
  Args_Opts Args;
  float * class_probs;
};

Avatar_handle* create_Avatar_handle(){
  //Allocate and zero out the struct, so that all pointers are NULL at first.
  //Then if any array fields go unused, we won't try to free them in cleanup.
  return calloc(1, sizeof(Avatar_handle));
}

// A cut-down version of the read_testing_data function from rw_data.c designed to only reread the test information and not do any re-initialization of the
// previously trained model
void read_testing_data_only(FC_Dataset *ds, CV_Class * Class, CV_Metadata train_meta, CV_Dataset *dataset, CV_Subset *subset, AV_SortedBlobArray *sorted_examples, Args_Opts *args) {
    int i;
    AV_ReturnCode rc;

    // Initialize some stuff
    if (args->format == EXODUS_FORMAT) {
        #ifdef HAVE_AVATAR_FCLIB
        // Read .classes file for exodus datasets
        read_classes_file(Class, args);
        // Let command line -V option override class file or set args value if not set
        if (args->class_var_name != NULL)
        {
            Class->class_var_name = av_strdup(args->class_var_name);
        }
        else
        {
            args->class_var_name = av_strdup(Class->class_var_name);
        }
        dataset->meta.num_classes =  Class->num_classes;
        dataset->meta.class_names = (char **)malloc(Class->num_classes * sizeof(char *));
        for (i = 0; i < Class->num_classes; i++)
            dataset->meta.class_names[i] = av_strdup(Class->class_names[i]);
        dataset->meta.num_fclib_seq = 0;
        #else
        av_printfErrorMessage("To use EXODUS_FORMAT, install the fclib 1.6.1 source in avatar/util/ and then rebuild.");
        #endif
    } 
    dataset->meta.exo_data.num_seq_meshes = 0;
    dataset->examples = (CV_Example*) calloc(1, sizeof(CV_Example));
    rc = av_initSortedBlobArray(sorted_examples);
    av_exitIfError(rc);

    if (args->format == EXODUS_FORMAT) {
        #ifdef HAVE_AVATAR_FCLIB
        // Add the testing data for the requested timesteps
        for (i = 0; i < args->num_test_times; i++) {
            //printf("Adding testing data for timestep %d\n", args->test_times[i]);
            if (! add_exo_data(*ds, args->test_times[i], dataset, Class, (i==0?TRUE:FALSE))) {
                fprintf(stderr, "Error adding data for timestep %d\n", args->test_times[i]);
                exit(-8);
            }
        }
        #else
        av_printfErrorMessage("To use EXODUS_FORMAT, install the fclib 1.6.1 source in avatar/util/ and then rebuild.");
        #endif
    } else if (args->format == AVATAR_FORMAT) {
      // FIXME: This need to be removed...
      // Calling this will generate a non-trivial books.  Why doesn't it do that the first time?
      /*
      if (! read_names_file(&dataset->meta, &Class, args, (args->do_training == TRUE ? FALSE : TRUE))) {
            fprintf(stderr, "Error reading names file\n");
            exit(-8);
            } 
      */
      // This guy gets kept
        if (! read_data_file(dataset, subset, Class, sorted_examples, "test", *args)) {
            fprintf(stderr, "Error reading data file\n");
            exit(-8);
        }
        subset->meta.exo_data.num_seq_meshes = 0;
    }
    
    // The functionality of this block is done in read_data_file() for non-exodus data
    if (args->format == EXODUS_FORMAT) {
        #ifdef HAVE_AVATAR_FCLIB
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
        #else
        av_printfErrorMessage("To use EXODUS_FORMAT, install the fclib 1.6.1 source in avatar/util/ and then rebuild.");
        #endif
    }
}


//Loads previously created avatar tree in order to test with new data
Avatar_handle* avatar_load(char* filestem, char* names_file, int names_file_is_a_string, char* trees_file, int trees_file_is_a_string) {
    //Create Avatar_handle object, then populate the handle with
    //the objects required to train and test avatar trees  
    Avatar_handle* a = create_Avatar_handle();
    int argc = 1;
    char* argv[] = {"./avatar_api.c"};

    a->Args = process_opts(argc, argv);
    a->Args.caller = AVATARDT_CALLER;

    int fslen = strlen(filestem);    
    a->Args.base_filestem = av_strdup(filestem);
    a->Args.names_file = (char*) malloc(sizeof(char)*(fslen + 20));
    sprintf(a->Args.names_file,"%s.names",filestem);
    a->Args.trees_file = (char*) malloc(sizeof(char)*(fslen + 20));
    sprintf(a->Args.trees_file,"%s.trees",filestem);
    a->Args.datafile = (char*) malloc(sizeof(char)*(fslen + 20));
    sprintf(a->Args.datafile,"%s.data",filestem);
    a->Args.data_path = av_strdup("");
  
    a->Args.format = AVATAR_FORMAT;

    //If the file_is_a_string flag is set for either
    //of the required file parameters, avatar will
    //read the file as a string 
    if (names_file_is_a_string > 0){
      a->Args.names_file_is_a_string = TRUE;
      a->Args.names_string = av_strdup(names_file);
    } 

    if (trees_file_is_a_string > 0){
      a->Args.trees_file_is_a_string = TRUE;
      a->Args.trees_string = av_strdup(trees_file);
    } 

    av_exitIfError(av_initSortedBlobArray(&a->Train_Sorted_Examples));

    a->Args.do_testing = TRUE;
    a->Args.do_training = FALSE;

    // Get all of the trees in
    read_names_file(&a->Train_Subset.meta, &a->Class, &a->Args, (a->Args.do_training == TRUE ? FALSE : TRUE));
    a->Test_Ensembles = calloc(1, sizeof(DT_Ensemble));
    read_ensemble(a->Test_Ensembles, -1, 0, &a->Args);

    // Make sure we allocate memory for testing
    free_CV_Class(a->Class); // Need to clean up class before it gets rewritten
    read_names_file(&a->Test_Dataset.meta, &a->Class, &a->Args, (a->Args.do_training == TRUE ? FALSE : TRUE));
    a->class_probs = (float*)malloc(a->Train_Subset.meta.num_classes * sizeof(float));
    return a;
}

//Takes arguments required for training Avatar decision trees, returns
//a stuct of type Avatar_handle that includes the required objects for 
//testing data
Avatar_handle* avatar_train(int argc, char ** argv, char* names_file, int names_file_is_a_string, char* train_file, int train_file_is_a_string){
    //Create Avatar_handle object, then populate the handle with
    //the objects required to train and test avatar trees  
    Avatar_handle* a = calloc(1, sizeof(Avatar_handle));

    #ifdef HAVE_AVATAR_FCLIB
    a->Test_Ensembles = calloc(1, sizeof(DT_Ensemble));

    a->Args = process_opts(argc, argv);
    a->Args.caller = AVATARDT_CALLER;
    char * filestem;
    filestem = av_strdup(a->Args.base_filestem);
    a->Args.names_file = strcat(filestem, ".names");
    filestem = av_strdup(a->Args.base_filestem);
    a->Args.trees_file = strcat(filestem, ".trees");

    a->Args.do_training = TRUE;

    //If the file_is_a_string flag is set for either
    //of the required file parameters, avatar will
    //read the file as a string 
    if (names_file_is_a_string > 0){
	a->Args.names_file_is_a_string = TRUE;
	a->Args.names_string = av_strdup(names_file);
    } 

    if (train_file_is_a_string > 0){
	a->Args.train_file_is_a_string = TRUE;
	a->Args.train_string = train_file;
    } 

    if (a->Args.format == EXODUS_FORMAT) {
            init_fc(a->Args);
            open_exo_datafile(&a->ds, a->Args.datafile);
        }

    //Read the required training data into the objects
    //stored in the Avatar_handle 
    //
    //
    //This initializes Train_Sorted_Examples.

    read_training_data(&a->ds, &a->Train_Dataset, &a->Train_Subset, &a->Train_Sorted_Examples, &a->Args);

    if (a->Args.num_minority_classes > 0)
        decode_minority_class_names(a->Train_Subset.meta, &a->Args);
    if (a->Args.do_smote)
        smote(&a->Train_Subset, &a->Train_Subset, &a->Train_Sorted_Examples, a->Args);
    if (a->Args.do_balanced_learning) 
        assign_bl_clump_numbers(a->Train_Subset.meta, a->Train_Sorted_Examples, a->Args);

    late_process_opts(a->Train_Dataset.meta.num_attributes, a->Train_Dataset.meta.num_examples, &a->Args);

    if (a->Args.do_ivote == TRUE || a->Args.do_bagging == TRUE)
        check_stopping_algorithm(1, 1, 0.0, 0, NULL, NULL, a->Args);
    display_dt_opts(stdout, "", a->Args, a->Train_Subset.meta);

    // Train 
        if (a->Args.do_ivote) {
            // If we're not testing now, create a stub testing subset
            if (! a->Args.do_testing)
                a->Test_Subset.meta.num_examples = 0;
            train_ivote(a->Train_Subset, a->Test_Subset, -1, &a->Cache, a->Args);
        } else {
            train(&a->Train_Subset, a->Test_Ensembles, -1, a->Args);
            // May need to re-read the ensemble for testing.
            // We'll delete the ensemble file later if --no-save-trees was requested
            //save_ensemble(Test_Ensembles[0], Train_Subset.meta, -1, Args, Train_Subset.meta.num_classes);
	    //free_DT_Ensemble(Train_Ensemble, TRAIN_MODE);
        }
    #else
    av_printfErrorMessage("To call avatar_train(), install the fclib 1.6.1 source in avatar/util/ and then rebuild.");
    #endif
    return a;
}

//Takes a previously created Avatar_handle in order to make predictions on 
//test data, returns an int array of predictions on the test data
void avatar_test(Avatar_handle* a, char* test_data_file, int test_data_is_a_string, int* predictions, float *probabilities){
    CV_Matrix Matrix = {0};
    a->Args.do_testing = TRUE;
    a->Args.do_training = FALSE;
    int i;
    int fslen = strlen(a->Args.base_filestem);
    free(a->Args.test_file);
    a->Args.test_file = (char*) malloc(sizeof(char)*(fslen + 20));
    sprintf(a->Args.test_file,"%s.test",a->Args.base_filestem);

    //If the file_is_a_string flag is set for either
    //of the required file parameters, avatar will
    //read the file as a string 
    if (test_data_is_a_string > 0){
      free(a->Args.test_string);
      a->Args.test_string = av_strdup(test_data_file);
      a->Args.test_file_is_a_string = TRUE;
    }

    //Trees string flag needs to be set because we will be 
    //reading the previously created tree from the handle,
    //which makes it a string
    a->Args.trees_file_is_a_string = TRUE;
    
    if (a->Args.format == EXODUS_FORMAT && ! a->Args.do_training) {
      #ifdef HAVE_AVATAR_FCLIB
      init_fc(a->Args);
      open_exo_datafile(&a->ds, a->Args.datafile);
      #else
      av_printfErrorMessage("To test using EXODUS_FORMAT, install the fclib 1.6.1 source in avatar/util/ and then rebuild.");
      #endif
    }

    //Read the required testing data into the objects 
    //stored in the Avatar_handle 
    av_freeSortedBlobArray(&a->Test_Sorted_Examples);
    read_testing_data_only(&a->ds, &a->Class, a->Train_Subset.meta, &a->Test_Dataset, &a->Test_Subset, &a->Test_Sorted_Examples, &a->Args);

    late_process_opts(a->Test_Dataset.meta.num_attributes, a->Test_Dataset.meta.num_examples, &a->Args);

    #ifdef HAVE_AVATAR_FCLIB
    if (a->Args.output_predictions)
      if (a->Args.format == EXODUS_FORMAT)
        init_predictions(a->Test_Subset.meta, &a->pred_prob, -1, a->Args);
    #endif

    //Create prediction matrix, then iterate over Test_Subset to get prediction 
    //for each sample in Test_Subset and return array of predictions	
    int num_ensembles = 1;
    
    if (num_ensembles == 1) {
      if (a->Args.do_boosting == TRUE){
        build_boost_prediction_matrix(a->Test_Subset, *a->Test_Ensembles, &Matrix);
      } else {
        build_prediction_matrix(a->Test_Subset, *a->Test_Ensembles, &Matrix);
      }
    } 
    else if (num_ensembles > 1) {
      DT_Ensemble big_ensemble;
      concat_ensembles(num_ensembles, a->Test_Ensembles, &big_ensemble);	      
      if (a->Args.do_boosting == TRUE){
        build_boost_prediction_matrix(a->Test_Subset, big_ensemble, &Matrix);		
      } else {
        build_prediction_matrix(a->Test_Subset, big_ensemble, &Matrix); 
      }
    }
        
    int line, leaf_node, class;
    
    // NOTE: This pointer grabs part of the interior of the tree.  This does not need to be free'd.
    float *this_probs = NULL;
    
    for (line = 0; line < a->Test_Subset.meta.num_examples; line++) {
      for (class = 0; class < a->Train_Subset.meta.num_classes; class++)
        a->class_probs[class] = 0;
      CV_Example e = a->Test_Subset.examples[line];
      e.predicted_class_num = find_best_class_from_matrix(line, Matrix, a->Args, line, 0);
      predictions[line] = e.predicted_class_num;
      for (i = 0; i < a->Test_Ensembles->num_trees; i++){
        this_probs = find_example_probabilities(a->Test_Ensembles->Trees[i], e, a->Test_Subset.float_data, &leaf_node);
        for (class = 0; class < a->Train_Subset.meta.num_classes; class++){
          a->class_probs[class] += this_probs[class]; 
        }
      }
      for (class = 0; class < a->Train_Subset.meta.num_classes; class++)
        a->class_probs[class] /= a->Test_Ensembles->num_trees;
      // Set each column to its corresponding prediction probability
      for (class = 0; class < a->Train_Subset.meta.num_classes; class++){
        probabilities[line * a->Train_Subset.meta.num_classes + class] = a->class_probs[class];
      }
    }
    for (i = 0; i < a->Test_Subset.meta.num_examples; i++)
      free(Matrix.data[i]);
    free(Matrix.data);
    free(Matrix.classes);
    free_CV_Subset(&a->Test_Subset, a->Args, TEST_MODE);
    av_freeSortedBlobArray(&a->Test_Sorted_Examples);
}


//Free memory allocated by the Avatar_handle
void avatar_cleanup(Avatar_handle* a){
  if(!a) return;
  free_DT_Ensemble(*a->Test_Ensembles, TEST_MODE);
  free(a->Test_Ensembles);
  //The following CV_Metadata members may be owned by Train_Subset and Test_Subset,
  //so free them if they are not aliasing the metadata for Train_Dataset and Test_Dataset
  free_CV_Metadata_Aliasing(&a->Train_Subset.meta, &a->Train_Dataset.meta, a->Args.format, a->Args.read_folds);
  free_CV_Metadata_Aliasing(&a->Test_Subset.meta, &a->Test_Dataset.meta, a->Args.format, a->Args.read_folds);
  free_CV_Subset(&a->Test_Subset, a->Args, TEST_MODE);
  free_CV_Subset_inter(&a->Train_Subset, a->Args, TEST_MODE); // is this right?
  memset(&a->Test_Subset, 0, sizeof(CV_Subset));
  memset(&a->Train_Subset, 0, sizeof(CV_Subset));

  free_CV_Dataset(a->Test_Dataset, a->Args);
  free_CV_Dataset(a->Train_Dataset, a->Args);
  memset(&a->Train_Dataset, 0, sizeof(CV_Dataset));
  memset(&a->Test_Dataset, 0, sizeof(CV_Dataset));

  av_freeSortedBlobArray(&a->Train_Sorted_Examples);
  av_freeSortedBlobArray(&a->Test_Sorted_Examples);
  
  free(a->class_probs);
  a->class_probs = NULL;
  free_CV_Class(a->Class);
  free_Args_Opts_Full(a->Args);
  free(a);
}

int avatar_num_classes(Avatar_handle* handle) {
  if(!handle) return -1;
  return handle->Class.num_classes;
}

