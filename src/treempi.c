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
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include "crossval.h"
#include "gain.h"
#include "array.h"
#include "util.h"
#include "memory.h"
#include "subspaces.h"
#include "bagging.h"
#include "evaluate.h"
#include "rw_data.h"
#include "tree.h"
#include "treempi.h"
#include "ivote.h"
#include "ivotempi.h"
#include "mpiL.h"
#include "options.h"
#include "heartbeat.h"
#include "reset.h"

//void _train_mpi(CV_Subset *data, DT_Ensemble *ensemble, int myrank, Args_Opts args) {}
void train_mpi(CV_Partition partitions, DT_Ensemble **ensemble, int myrank, Args_Opts args) {
    int i, j;
    int num_procs;
    int num_parts = 0;
    MPI_Status status;
    CV_Subset *train_data;
    FC_Dataset ds;
    Vote_Cache *cache;
    FILE *oob_file = NULL;
    char **oob_filename;
    
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int *proc_is_crunching;
    int *part_is_being_crunched_on;
    Boolean *part_is_complete;
    int last_part_assigned = -1;
    proc_is_crunching = (int *)malloc(num_procs * sizeof(int));
    
    // Compute OOB accuracy if using stopping algorithm or if bagging
    // Since stopping algorithm must be used with bagging or ivoting, then compute if bagging
    Boolean compute_oob_acc = args.bag_size > 0.0 ? TRUE : FALSE;
    
    //_broadcast_subset(data, myrank, args);
    if (myrank == 0) {
        num_parts = partitions.num_partitions;
        part_is_being_crunched_on = (int *)malloc(num_parts * sizeof(int));
        part_is_complete = (Boolean *)malloc(num_parts * sizeof(Boolean));
        oob_filename = (char **)calloc(num_parts, sizeof(char *));
        cache = (Vote_Cache *)malloc(num_parts * sizeof(Vote_Cache));
        //last_part_assigned = -1;
        for (i = 0; i < num_procs; i++)
            proc_is_crunching[i] = -1;
        for (i = 0; i < num_parts; i++) {
            part_is_being_crunched_on[i] = -1;
            part_is_complete[i] = FALSE;
        }

        train_data = (CV_Subset *)malloc(num_parts * sizeof(CV_Subset));
        //printf("Getting ready to read training data and send first batch\n");
        // Read num_procs of the partitions and send them off for processing.
        // Loop around if num_parts < num_procs
        for (i = 1; i < num_procs; i++) {
            //last_part_assigned++;
            last_part_assigned = (last_part_assigned+1) % num_parts;
            reset_CV_Subset(&train_data[last_part_assigned]);
            if (part_is_being_crunched_on[last_part_assigned] == -1) {
                CV_Dataset Train_Dataset;
                reset_CV_Metadata(&Train_Dataset.meta);
                AV_SortedBlobArray Train_Sorted_Examples;
                if (args.format == EXODUS_FORMAT)
                    open_exo_datafile(&ds, partitions.partition_datafile[last_part_assigned]);
                read_training_data(&ds, &Train_Dataset, &train_data[last_part_assigned], &Train_Sorted_Examples, &args);
                reset_DT_Ensemble(& ((*ensemble)[last_part_assigned]));
                (*ensemble)[last_part_assigned].num_trees = 100;
                init_ensemble(&(*ensemble)[last_part_assigned], train_data[last_part_assigned].meta, args);
                if (compute_oob_acc == TRUE) {
                    initialize_cache(&cache[last_part_assigned], train_data[last_part_assigned].meta.num_classes,
                                     train_data[last_part_assigned].meta.num_examples, 0, 0);
                }
                // Initialize the oob-data file for --verbose-oob
                if (args.output_verbose_oob == TRUE)
                    oob_filename[last_part_assigned] = av_strdup(write_tree_file_header(args.num_trees,
                                                       train_data[last_part_assigned].meta, -1, args.oob_file, args));
            }
            send_subset(&train_data[last_part_assigned], i, args);
            proc_is_crunching[i] = last_part_assigned;
            part_is_being_crunched_on[last_part_assigned] = i;
            // I think there are some pointer copies from CV_Dataset to CV_Subset
            // so freeing the Train_Dataset will destroy some needed info in train_data
            // But, with Train_Dataset inside if{} block above, isn't it destroyed
            // when block is exited?
            //free_CV_Dataset(Train_Dataset, args);
        }
    } else {
        train_data = (CV_Subset *)malloc(sizeof(CV_Subset));
        receive_subset(&train_data[0], myrank, args);
    }
    
    //printf("rank %d pid = %d\n", myrank, getpid());

    CV_Subset *data_rs, *data_bag;
    data_bag = (CV_Subset *)malloc(sizeof(CV_Subset));
    data_rs = (CV_Subset *)malloc(sizeof(CV_Subset));
    reset_CV_Subset(&data_bag[0]);

    // Initialize weights to flat
    //data->weights = (float *)malloc(data->num_examples * sizeof(float));
    //for (i = 0; i < data->num_examples; i++)
    //    data->weights[i] = 1.0/data->num_examples;
    
    if (myrank == 0) {
        begin_progress_counters(num_parts);
        
        int **thist;
        thist = (int **)malloc(num_parts * sizeof(int *));
        for (i = 0; i < num_parts; i++)
            thist[i] = (int *)calloc(num_procs, sizeof(int));
        int num_procs_signaled = 0;
        int *num_trees_collected;
        num_trees_collected = (int *)calloc(num_parts, sizeof(int));
        int this_pack_num;
        int position;
        int *num_nodes;
        int packsize = AT_MPI_BUFMAX;
        int tree_passing_code;
        char *buff;
        buff = (char *)malloc(AT_MPI_BUFMAX * sizeof(char));
        int *in_bag;
        int this_proc; // Rank we are communicating with right now
        int this_part; // Partition that this_proc is crunching right now
        
        int *stop_building_at;
        stop_building_at = (int *)calloc(num_parts, sizeof(int));
        float *best_oob_acc;
        best_oob_acc = (float *)calloc(num_parts, sizeof(float));
        
        // Loop until all MPI processes have been told to stop building trees
        while (num_procs_signaled < num_procs-1) {
            
            // Receive a message asking for permission to send a tree
            MPI_Recv(buff, AT_MPI_BUFMAX, MPI_INT, MPI_ANY_SOURCE, MPI_PERM2SEND_TAG, MPI_COMM_WORLD, &status);
            this_proc = status.MPI_SOURCE;
            this_part = proc_is_crunching[this_proc];
            
            if ( (args.num_trees > 0 && num_trees_collected[this_part] < args.num_trees) ||
                 (args.auto_stop == TRUE && stop_building_at[this_part] == 0) ) {
                
                // If we need more trees, send "yes" and then receive tree
                tree_passing_code = AT_SEND_ONE_TREE;
                MPI_Ssend(&tree_passing_code, 1, MPI_INT, this_proc, MPI_PERM2SEND_TAG, MPI_COMM_WORLD);
                //printf("\nReceiving tree from %d\n", this_proc);
                MPI_Recv(buff, AT_MPI_BUFMAX, MPI_PACKED, MPI_ANY_SOURCE, MPI_TREENODE_TAG, MPI_COMM_WORLD, &status);
                position = 0;
                
                // Get the number of trees in this pack
                MPI_Unpack(buff, packsize, &position, &this_pack_num, 1, MPI_INT, MPI_COMM_WORLD);
                //printf("Receiving %d trees\n", this_pack_num);
                num_nodes = (int *)calloc(this_pack_num, sizeof(int));
                // Get number of nodes in each tree
                for (i = 0; i < this_pack_num; i++)
                    MPI_Unpack(buff, packsize, &position, &num_nodes[i], 1, MPI_INT, MPI_COMM_WORLD);
                for (i = 0; i < this_pack_num; i++) {
                    thist[this_part][this_proc]++;
                    //printf("Receiving tree %d with %d nodes from %d\n", i+num_trees_collected, num_nodes[i], this_proc);
                    receive_one_tree(buff, &position, &(*ensemble)[this_part].Trees[i+num_trees_collected[this_part]],
                                     train_data[0].meta.num_classes, num_nodes[i]);
                    //printf("Done receiving tree\n");
                }
                if (compute_oob_acc == TRUE) {
                    // Receive in_bag info
                    in_bag = (int *)malloc(train_data[this_part].meta.num_examples * sizeof(int));
                    //printf("Rank 0 is receiving the in_bag (%d) info from rank %d\n", data->num_examples, this_proc);
                    MPI_Recv(in_bag, train_data[this_part].meta.num_examples, MPI_INT, this_proc, MPI_VOTE_CACHE_TAG,
                             MPI_COMM_WORLD, &status);
                    //printf("Rank 0 got the in_bag info\n");
                    for (i = 0; i < train_data[this_part].meta.num_examples; i++)
                        train_data[this_part].examples[i].in_bag = in_bag[i];
                    free(in_bag);
                    
                    //printf("Rank 0 is updating the Vote_Cache\n");
                    cache[this_part].current_classifier_count = num_trees_collected[this_part] + 1;
                    
                    // Update Vote_Cache
                    cache[this_part].oob_error =
                                    compute_oob_error_rate((*ensemble)[this_part].Trees[num_trees_collected[this_part]],
                                                           train_data[this_part], &cache[this_part], args);
                    //printf("Err for part %d = %g\n", this_part, unweighted_oob_error);
                    // Check stopping algorithm
                    stop_building_at[this_part] =
                           check_stopping_algorithm(0, this_part, 1.0 - cache[this_part].oob_error,
                                                    num_trees_collected[this_part]+1, &best_oob_acc[this_part],
                                                    oob_filename[this_part], args);
                    //if (args.output_verbose_oob) {
                    //    if ((oob_file = fopen(oob_filename[this_part], "a")) == NULL) {
                    //        fprintf(stderr, "Failed to open oob file for saving oob accuracies: '%s'\nExiting ...\n",
                    //                        oob_filename[this_part]);
                    //        exit(8);
                    //    }
                    //    fprintf(oob_file, "%d %6g\n", num_trees_collected[this_part]+1, 1-cache->oob_error);
                    //}
                    if (args.auto_stop == TRUE && stop_building_at[this_part] > 0) {
                        (*ensemble)[this_part].num_trees = stop_building_at[this_part];
                        //printf("Stopping with %d trees\n", ensemble->num_trees);
                        if (args.output_verbose_oob) {
                            if ((oob_file = fopen(oob_filename[this_part], "a")) == NULL) {
                                fprintf(stderr, "Failed to open oob file for saving oob accuracies: '%s'\nExiting ...\n",
                                                oob_filename[this_part]);
                                exit(8);
                            }
                            fprintf(oob_file, "#Stopping with %d trees\n", (*ensemble)[this_part].num_trees);
                            fclose(oob_file);
                        }
                    }
                }
                
                // REVIEW-2012-03-10-ArtM: wait, can't multiple trees
                // be received in the for loop above?
                num_trees_collected[this_part]++;
            } else {
                // This partition is done.
                
                // Mark as complete
                part_is_complete[this_part] = TRUE;
                
                // Look for a partition that is not complete
                // Start with the next partition after the one most recently assigned to effect a round-robin scheme
                int keep_going_with = (last_part_assigned+1) % num_parts;
                while (keep_going_with != last_part_assigned && part_is_complete[keep_going_with] == TRUE)
                    keep_going_with = (keep_going_with+1) % num_parts;
                // We can get back to the same partition as the last one assigned and it may not be complete
                // Only cease and desist if we get back to the last assigned and it IS complete
                if (keep_going_with == last_part_assigned && part_is_complete[keep_going_with] == TRUE) {
                    // Signal this process to cease and desist
                    tree_passing_code = AT_SEND_NO_TREE;
                    MPI_Ssend(&tree_passing_code, 1, MPI_INT, this_proc, MPI_PERM2SEND_TAG, MPI_COMM_WORLD);
                    num_procs_signaled++;
                } else {
                    last_part_assigned = keep_going_with;
                    tree_passing_code = AT_RECEIVE_NEW_DATA;
                    MPI_Ssend(&tree_passing_code, 1, MPI_INT, this_proc, MPI_PERM2SEND_TAG, MPI_COMM_WORLD);
                    //printf("Reading part %d (%s) and send to proc %d\n",
                    //       keep_going_with, partitions.partition_filenames[keep_going_with], this_proc);
                    if (part_is_being_crunched_on[keep_going_with] == -1) {
                        CV_Dataset Train_Dataset;
                        AV_SortedBlobArray Train_Sorted_Examples;
                        if (args.format == EXODUS_FORMAT)
                            open_exo_datafile(&ds, partitions.partition_datafile[keep_going_with]);
                        read_training_data(&ds, &Train_Dataset, &train_data[keep_going_with],
                                           &Train_Sorted_Examples, &args);
                        init_ensemble(&(*ensemble)[keep_going_with], train_data[keep_going_with].meta, args);
                        if (compute_oob_acc == TRUE) {
                            initialize_cache(&cache[keep_going_with], train_data[keep_going_with].meta.num_classes,
                                             train_data[keep_going_with].meta.num_examples, 0, 0);
                        }
                    }
                    send_subset(&train_data[keep_going_with], this_proc, args);
                    proc_is_crunching[this_proc] = keep_going_with;
                    part_is_being_crunched_on[keep_going_with] = this_proc;
                }
            }

            update_progress_counters(num_parts, num_trees_collected);
        }
        end_progress_counters();

        if (args.auto_stop == TRUE) {
            for (i = 0; i < num_parts; i++) {
                printf("Partition %d Stopping Algorithm Result: %d trees with an oob accuracy of %.4f%%\n",
                        i, stop_building_at[i], best_oob_acc[i] * 100.0);
            }
        } else if (compute_oob_acc == TRUE) {
            for (i = 0; i < num_parts; i++) {
                printf("Partition %d: oob accuracy = %.4f%%\n", i, (1.0 - cache[i].oob_error) * 100.0);
            }
        }
        if (args.show_per_process_stats == TRUE) {
            printf("Per Partition Per Process Stats:\n");
            for (i = 0; i < num_parts; i++) {
                printf(" Partition %d:\n", i);
                printf("  Number of Trees:\n");
                for (j = 1; j < num_procs; j++)
                    printf("    Proc %d: %5d(%5.1f%%)\n",
                           j, thist[i][j], 100.0*(float)thist[i][j]/(float)num_trees_collected[i]);
                printf("  Average: %.2f\n", int_average(thist[i], 1, num_procs-1));
                printf("  Std Dev: %.4f\n", int_stddev(thist[i], 1, num_procs-1));
                printf("\n");
            }
        }
        free(buff);
        //printf("Rank 0 is continuing\n");
        
    } else { // Here, myrank != 0
        int keep_going = AT_SEND_ONE_TREE;
        int *in_bag;
        
        while (keep_going != AT_SEND_NO_TREE) {
            (*ensemble)[0].num_trees = 1;
            (*ensemble)[0].Trees = (DT_Node **)malloc(1 * sizeof(DT_Node *));
            (*ensemble)[0].Books = (Tree_Bookkeeping *)malloc(1 * sizeof(Tree_Bookkeeping));
            
            if (args.do_bagging == TRUE)
                make_bag(&train_data[0], data_bag, args, 0);
            else
                data_bag = &train_data[0];
            
            if (args.random_subspaces > 0)
                apply_random_subspaces(*data_bag, data_rs, args);
            else
                data_rs = data_bag;
            
            // Initialize the array of DT_Nodes for this ensemble and some other values
            (*ensemble)[0].Books[0].num_malloced_nodes = 1;
            (*ensemble)[0].Books[0].next_unused_node = 1;
            (*ensemble)[0].Books[0].current_node = 0;
            (*ensemble)[0].Trees[0] = (DT_Node *)malloc((*ensemble)[0].Books[0].num_malloced_nodes * sizeof(DT_Node));
            build_tree(data_rs, &(*ensemble)[0].Trees[0], &(*ensemble)[0].Books[0], args);
            
            if (args.do_bagging == TRUE) {
                for (j = 0; j < (int)(args.bag_size * (float)data_bag->meta.num_examples / 100.0); j++)
                    free(data_bag->examples[j].distinct_attribute_values);
                free_CV_Subset_inter(data_bag, args, TRAIN_MODE);
            }
            if (args.random_subspaces > 0)
                free_CV_Subset_inter(data_rs, args, TRAIN_MODE);
            
            // Ask if more trees are needed
            MPI_Ssend(&keep_going, 1, MPI_INT, 0, MPI_PERM2SEND_TAG, MPI_COMM_WORLD);
            MPI_Recv(&keep_going, 1, MPI_INT, 0, MPI_PERM2SEND_TAG, MPI_COMM_WORLD, &status);
            if (keep_going == AT_SEND_ONE_TREE) {
                //printf("Proc %d got the sugnal to send one tree\n", myrank);
                send_trees((*ensemble)[0].Trees, (*ensemble)[0].Books, train_data[0].meta.num_classes, 1);
                if (compute_oob_acc == TRUE) {
                    // Send array of in_bag values
                    in_bag = (int *)malloc(train_data[0].meta.num_examples * sizeof(int));
                    for (i = 0; i < train_data[0].meta.num_examples; i++)
                        in_bag[i] = train_data[0].examples[i].in_bag;
                    //printf("Rank %d is sending the in_bag (%d) info now\n", myrank, train_data.num_examples);
                    MPI_Ssend(in_bag, train_data[0].meta.num_examples, MPI_INT, 0, MPI_VOTE_CACHE_TAG, MPI_COMM_WORLD);
                    free(in_bag);
                }
            } else if (keep_going == AT_RECEIVE_NEW_DATA) {
                //printf("Proc %d for the signal to receive new data\n", myrank);
                receive_subset(&train_data[0], myrank, args);
            } else {
                //printf("Proc %d got the signal to cease and desist\n", myrank);
            }
            free_DT_Ensemble((*ensemble)[0], TRAIN_MODE);
            //if (myrank == 1) sleep(1);
        }
        //printf("Rank %d is done building trees\n", myrank);
    }

}

//void _train_ivote_mpi(CV_Subset train_data, CV_Subset test_data, int fold_num, Vote_Cache *cache, int myrank, Args_Opts args) {}
//Modified by DACIESL June-04-08: Laplacean Estimates
//uses new save_tree call
void train_ivote_mpi(int fold_num, CV_Partition partitions, int myrank, Args_Opts args) {
    int i, j;
    int num_procs, num_parts;
    MPI_Status status;
    CV_Subset *train_data;
    FC_Dataset ds;
    Vote_Cache *Cache; 
    FILE *oob_file = NULL;
    char **oob_filename;
    
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int *proc_is_crunching;
    int *part_is_being_crunched_on;
    Boolean *part_is_complete;
    int last_part_assigned = -1;
    proc_is_crunching = (int *)malloc(num_procs * sizeof(int));
    
    // Compute OOB accuracy if using stopping algorithm or if bagging
    // Since stopping algorithm must be used with bagging or ivoting, then compute if bagging
    Boolean compute_oob_acc = TRUE;
    
    //broadcast_subset(&train_data, myrank, args);
    //broadcast_subset(&test_data, myrank, args);
    if (myrank == 0) {
        num_parts = partitions.num_partitions;
        part_is_being_crunched_on = (int *)malloc(num_parts * sizeof(int));
        part_is_complete = (Boolean *)malloc(num_parts * sizeof(Boolean));
        oob_filename = (char **)calloc(num_parts, sizeof(char *));
        Cache = (Vote_Cache *)malloc(num_parts * sizeof(Vote_Cache));
        //last_part_assigned = -1;
        for (i = 0; i < num_procs; i++)
            proc_is_crunching[i] = -1;
        for (i = 0; i < num_parts; i++) {
            part_is_being_crunched_on[i] = -1;
            part_is_complete[i] = FALSE;
        }
        
        train_data = (CV_Subset *)malloc(num_parts * sizeof(CV_Subset));
        //printf("Getting ready to read training data and send first batch\n");
        // Read num_procs of the partitions and send them off for processing.
        // Loop around if num_parts < num_procs
        for (i = 1; i < num_procs; i++) {
            last_part_assigned = (last_part_assigned+1) % num_parts;
            //printf("Reading part %d (%s) and send to proc %d\n",
            //       last_part_assigned, partitions.partition_filenames[last_part_assigned], i);
            if (part_is_being_crunched_on[last_part_assigned] == -1) {
                CV_Dataset Train_Dataset;
                AV_SortedBlobArray Train_Sorted_Examples;
                if (args.format == EXODUS_FORMAT)
                    open_exo_datafile(&ds, partitions.partition_datafile[last_part_assigned]);
                read_training_data(&ds, &Train_Dataset, &train_data[last_part_assigned], &Train_Sorted_Examples, &args);
                init_ensemble_file(train_data[last_part_assigned].meta, partitions, last_part_assigned, fold_num, args);
                initialize_cache(&Cache[last_part_assigned], train_data[last_part_assigned].meta.num_classes,
                                 train_data[last_part_assigned].meta.num_examples, 0, 0);
                // Initialize the oob-data file for --verbose-oob
                if (args.output_verbose_oob == TRUE)
                    oob_filename[last_part_assigned] = av_strdup(write_tree_file_header(args.num_trees,
                                                       train_data[last_part_assigned].meta, -1, args.oob_file, args));
            }
            send_subset(&train_data[last_part_assigned], i, args);
            proc_is_crunching[i] = last_part_assigned;
            part_is_being_crunched_on[last_part_assigned] = i;
            // I think there are some pointer copies from CV_Dataset to CV_Subset
            // so freeing the Train_Dataset will destroy some needed info in train_data
            // But, with Train_Dataset inside if{} block above, isn't it destroyed
            // when block is exited?
            //free_CV_Dataset(Train_Dataset, args);
        }
    } else {
        train_data = (CV_Subset *)malloc(sizeof(CV_Subset));
        receive_subset(&train_data[0], myrank, args);
    }
    
    if (myrank == 0) {
        begin_progress_counters(num_parts);
        
        int **thist;
        thist = (int **)malloc(num_parts * sizeof(int *));
        for (i = 0; i < num_parts; i++)
            thist[i] = (int *)calloc(num_procs, sizeof(int));
        int num_procs_signaled = 0;
        int *num_trees_collected;
        num_trees_collected = (int *)calloc(num_parts, sizeof(int));
        int this_pack_num;
        int position;
        int *num_nodes;
        int packsize = AT_MPI_BUFMAX;
        int tree_passing_code;
        char *buff;
        buff = (char *)malloc(AT_MPI_BUFMAX * sizeof(char));
        int *in_bag;
        int this_proc; // Rank we are communicating with right now
        int this_part; // Partition that this_proc is crunching right now
        
        DT_Node *Tree;
        
        int *stop_building_at;
        stop_building_at = (int *)calloc(num_parts, sizeof(int));
        float *best_oob_acc;
        best_oob_acc = (float *)malloc(num_parts * sizeof(float));
        float *unweighted_oob_error;
        unweighted_oob_error = (float *)malloc(num_parts * sizeof(float));
        
        while (num_procs_signaled < num_procs-1) {
            
            // Receive a message asking for permission to send a tree
            //printf("Rank 0 is being petitioned to receive a tree\n");
            MPI_Recv(buff, AT_MPI_BUFMAX, MPI_INT, MPI_ANY_SOURCE, MPI_PERM2SEND_TAG, MPI_COMM_WORLD, &status);
            this_proc = status.MPI_SOURCE;
            this_part = proc_is_crunching[this_proc];
            
            if ( (args.num_trees > 0 && num_trees_collected[this_part] < args.num_trees) ||
                 (args.auto_stop == TRUE && stop_building_at[this_part] == 0) ) {
                
                // If we need more trees, send "yes" and then receive tree
                //printf("Rank 0 is responding in the affirmative to rank %d\n", this_proc);
                tree_passing_code = AT_SEND_ONE_TREE;
                MPI_Ssend(&tree_passing_code, 1, MPI_INT, this_proc, MPI_PERM2SEND_TAG, MPI_COMM_WORLD);
                //printf("Rank 0 is receiving a tree now from rank %d\n", this_proc);
                fflush(NULL);
                MPI_Recv(buff, AT_MPI_BUFMAX, MPI_PACKED, MPI_ANY_SOURCE, MPI_TREENODE_TAG, MPI_COMM_WORLD, &status);
                position = 0;
                
                // Get the number of trees in this pack
                MPI_Unpack(buff, packsize, &position, &this_pack_num, 1, MPI_INT, MPI_COMM_WORLD);
                num_nodes = (int *)calloc(this_pack_num, sizeof(int));
                // Get number of nodes in each tree and then get all trees
                for (i = 0; i < this_pack_num; i++)
                    MPI_Unpack(buff, packsize, &position, &num_nodes[i], 1, MPI_INT, MPI_COMM_WORLD);
                for (i = 0; i < this_pack_num; i++) {
                    //printf("Rank 0 is unpacking a tree now from rank %d with %d nodes\n", this_proc, num_nodes[i]);
                    thist[this_part][this_proc]++;
                    receive_one_tree(buff, &position, &Tree, train_data[0].meta.num_classes, num_nodes[i]);
                }
                // Receive in_bag info
                in_bag = (int *)malloc(train_data[this_part].meta.num_examples * sizeof(int));
                //printf("Rank 0 is receiving the in_bag (%d) info from rank %d\n", train_data.num_examples, this_proc);
                MPI_Recv(in_bag, train_data[this_part].meta.num_examples, MPI_INT, this_proc, MPI_VOTE_CACHE_TAG, MPI_COMM_WORLD, &status);
                //printf("Rank 0 got the in_bag info\n");
                for (i = 0; i < train_data[this_part].meta.num_examples; i++)
                    train_data[this_part].examples[i].in_bag = in_bag[i];
                free(in_bag);
                
                // Send current oob_error and best_train_class from Vote_Cache
                MPI_Ssend(&Cache[this_part].oob_error, 1, MPI_DOUBLE, this_proc, MPI_VOTE_CACHE_TAG, MPI_COMM_WORLD);
                MPI_Ssend(Cache[this_part].best_train_class, train_data[this_part].meta.num_examples, MPI_INT, this_proc,
                         MPI_VOTE_CACHE_TAG, MPI_COMM_WORLD);

                //printf("Rank 0 is updating the Vote_Cache\n");
                Cache[this_part].current_classifier_count = num_trees_collected[this_part] + 1;
                // Update Vote_Cache
                unweighted_oob_error[this_part] = compute_oob_error_rate(Tree, train_data[this_part], &Cache[this_part], args);
                // Check stopping algorithm
                if (compute_oob_acc == TRUE) {
                    stop_building_at[this_part] = check_stopping_algorithm(0, this_part, 1.0 - unweighted_oob_error[this_part],
                                                                           num_trees_collected[this_part]+1,
                                                                           &best_oob_acc[this_part],
                                                                           oob_filename[this_part], args);
                    //if (args.output_verbose_oob) {
                    //    if ((oob_file = fopen(oob_filename[this_part], "a")) == NULL) {
                    //        fprintf(stderr, "Failed to open oob file for saving oob accuracies: '%s'\nExiting ...\n",
                    //                        oob_filename[this_part]);
                    //        exit(8);
                    //    }
                    //    fprintf(oob_file, "%d %6g\n", num_trees_collected[this_part]+1, 1.0 - unweighted_oob_error[this_part]);
                    //}
                    if (args.auto_stop == TRUE && stop_building_at[this_part] > 0) {
                        Cache[this_part].num_classifiers = stop_building_at[this_part];
                        //printf("Stopping with %d trees\n", Cache[this_part].num_classifiers);
                        if (args.output_verbose_oob) {
                            if ((oob_file = fopen(oob_filename[this_part], "a")) == NULL) {
                                fprintf(stderr, "Failed to open oob file for saving oob accuracies: '%s'\nExiting ...\n",
                                                oob_filename[this_part]);
                                exit(8);
                            }
                            fprintf(oob_file, "#Stopping with %d trees\n", Cache[this_part].num_classifiers);
                            fclose(oob_file);
                        }
                    }
                }
                Cache[this_part].oob_error = args.ivote_p_factor * Cache[this_part].oob_error +
                                   (1.0 - args.ivote_p_factor) * unweighted_oob_error[this_part];

                if (args.save_trees) {
                    args.datafile = av_strdup(partitions.partition_datafile[this_part]);
                    args.base_filestem = av_strdup(partitions.partition_base_filestem[this_part]);
                    args.data_path = av_strdup(partitions.partition_data_path[this_part]);
                    set_output_filenames(&args, FALSE, TRUE);
                    save_tree(Tree, fold_num, num_trees_collected[this_part]+1, args, train_data[this_part].meta.num_classes);
                }
                
                // REVIEW-2012-03-10-ArtM: wait, can't multiple trees
                // be received in the for loop above?
                num_trees_collected[this_part]++;
                
            } else {
                // This partition is done.
                
                // Mark as complete
                part_is_complete[this_part] = TRUE;
                
                // Look for a partition that is not complete
                // Start with the next partition after the one most recently assigned to effect a round-robin scheme
                int keep_going_with = (last_part_assigned+1) % num_parts;
                while (keep_going_with != last_part_assigned && part_is_complete[keep_going_with] == TRUE)
                    keep_going_with = (keep_going_with+1) % num_parts;
                // We can get back to the same partition as the last one assigned and it may not be complete
                // Only cease and desist if we get back to the last assigned and it IS complete
                if (keep_going_with == last_part_assigned && part_is_complete[keep_going_with] == TRUE) {
                    // Signal this process to cease and desist
                    tree_passing_code = AT_SEND_NO_TREE;
                    MPI_Ssend(&tree_passing_code, 1, MPI_INT, this_proc, MPI_PERM2SEND_TAG, MPI_COMM_WORLD);
                    num_procs_signaled++;
                } else {
                    last_part_assigned = keep_going_with;
                    tree_passing_code = AT_RECEIVE_NEW_DATA;
                    MPI_Ssend(&tree_passing_code, 1, MPI_INT, this_proc, MPI_PERM2SEND_TAG, MPI_COMM_WORLD);
                    //printf("Reading part %d (%s) and send to proc %d\n",
                    //       keep_going_with, partitions.partition_filenames[keep_going_with], this_proc);
                    if (part_is_being_crunched_on[keep_going_with] == -1) {
                        CV_Dataset Train_Dataset;
                        AV_SortedBlobArray Train_Sorted_Examples;
                        if (args.format == EXODUS_FORMAT)
                            open_exo_datafile(&ds, partitions.partition_datafile[keep_going_with]);
                        read_training_data(&ds, &Train_Dataset, &train_data[keep_going_with],
                                           &Train_Sorted_Examples, &args);
                        init_ensemble_file(train_data[keep_going_with].meta, partitions, keep_going_with,
                                           fold_num, args);
                        initialize_cache(&Cache[keep_going_with], train_data[keep_going_with].meta.num_classes,
                                         train_data[keep_going_with].meta.num_examples, 0, 0);
                    }
                    send_subset(&train_data[keep_going_with], this_proc, args);
                    proc_is_crunching[this_proc] = keep_going_with;
                    part_is_being_crunched_on[keep_going_with] = this_proc;
                }
            }

            update_progress_counters(num_parts, num_trees_collected);
        }
        end_progress_counters();

        if (compute_oob_acc == TRUE) {
            if (args.auto_stop == TRUE) {
                for (i = 0; i < num_parts; i++) {
                    printf("Partition %d Stopping Algorithm Result: %d trees with an oob accuracy of %.4f%%\n",
                            i, stop_building_at[i], best_oob_acc[i] * 100);
                    if (args.save_trees) {
                        // Read and rewrite ensemble file to get NumTrees right and eliminate extra trees
                        DT_Ensemble ensemble;
                        args.datafile = av_strdup(partitions.partition_datafile[i]);
                        args.base_filestem = av_strdup(partitions.partition_base_filestem[i]);
                        args.data_path = av_strdup(partitions.partition_data_path[i]);
                        set_output_filenames(&args, FALSE, TRUE);
                        read_ensemble(&ensemble, -1, stop_building_at[i], &args);
                        save_ensemble(ensemble, train_data[i].meta, fold_num, args, train_data[i].meta.num_classes);
                    }
                }
            } else {
                for (i = 0; i < num_parts; i++) {
                    printf("Partition %d: oob accuracy of %.4f%%\n", i, (1.0 - unweighted_oob_error[i]) * 100.0);
                }
            }
        }
        
        if (args.show_per_process_stats == TRUE) {
            printf("Per Partition Per Process Stats:\n");
            for (i = 0; i < num_parts; i++) {
                printf(" Partition %d:\n", i);
                printf("  Number of Trees:\n");
                for (j = 1; j < num_procs; j++)
                    printf("    Proc %d: %5d(%5.1f%%)\n",
                           j, thist[i][j], 100.0*(float)thist[i][j]/(float)num_trees_collected[i]);
                printf("  Average: %.2f\n", int_average(thist[i], 1, num_procs-1));
                printf("  Std Dev: %.4f\n", int_stddev(thist[i], 1, num_procs-1));
                printf("\n");
            }
        }

        free(buff);
    } else {
        int keep_going = AT_SEND_ONE_TREE;
        CV_Subset data_bite, *data_rs;
        Tree_Bookkeeping Books;
        DT_Node *Tree;
        int *in_bag;
        double oob_error = 0.0;
        int *best_train_class;
        best_train_class = (int *)malloc(train_data[0].meta.num_examples * sizeof(int));
        for (i = 0; i < train_data[0].meta.num_examples; i++)
            best_train_class[i] = -1;
        
        //int num_trees = 0;
        //printf("Rank %d starting  up\n", myrank);
        while (keep_going != AT_SEND_NO_TREE) {
            
            //Cache->current_classifier_count = num_trees + 1;
            
            //make_bite(&train_data[0], &data_bite, Cache, args);
            //printf("Rank %d picks %ld in loop 1\n", myrank, lrand48());
            make_bite_mpi(&train_data[0], &data_bite, oob_error, best_train_class, args);
            //printf("data_bite.high(b) is at memory location 0x%lx\n", &data_bite.high);
            if (args.random_subspaces > 0) {
                apply_random_subspaces(data_bite, data_rs, args);
                //printf("data_bite.high(m) is at memory location 0x%lx\n", &data_bite.high);
                //printf("data_rs.high(m)   is at memory location 0x%lx\n", &(data_rs->high));
            } else {
                data_rs = &data_bite;
            }
            
            // Initialize the array of DT_Nodes for this ensemble and some other values
            Books.num_malloced_nodes = 1;
            Books.next_unused_node = 1;
            Books.current_node = 0;
            Tree = (DT_Node *)malloc(Books.num_malloced_nodes * sizeof(DT_Node));
            
            //printf("Rank %d is building a tree now\n", myrank);
            //printf("Rank %d picks %ld in loop 2\n", myrank, lrand48());
            build_tree(&data_bite, &Tree, &Books, args);
            
            if (args.random_subspaces > 0) {
                free_CV_Subset_inter(data_rs, args, TRAIN_MODE);
            }
            
            for (i = 0; i < args.bite_size; i++)
                free(data_bite.examples[i].distinct_attribute_values);
            free_CV_Subset_inter(&data_bite, args, TRAIN_MODE);
            //free(data_bite.high);
            
            // Ask if more trees are needed
            //printf("Rank %d is asking to send a tree\n", myrank);
            MPI_Ssend(&keep_going, 1, MPI_INT, 0, MPI_PERM2SEND_TAG, MPI_COMM_WORLD);
            MPI_Recv(&keep_going, 1, MPI_INT, 0, MPI_PERM2SEND_TAG, MPI_COMM_WORLD, &status);
            //printf("Rank %d got %d from rank %d with error %d\n", myrank, keep_going, status.MPI_SOURCE, status.MPI_ERROR);
            if (keep_going == AT_SEND_ONE_TREE) {
                //printf("Rank %d is sending a tree now\n", myrank);
                // Send tree if needed
                send_trees(&Tree, &Books, train_data[0].meta.num_classes, 1);
                // Send array of in_bag values
                in_bag = (int *)malloc(train_data[0].meta.num_examples * sizeof(int));
                for (i = 0; i < train_data[0].meta.num_examples; i++)
                    in_bag[i] = train_data[0].examples[i].in_bag;
                //printf("Rank %d is sending the in_bag (%d) info now\n", myrank, train_data[0].meta.num_examples);
                MPI_Ssend(in_bag, train_data[0].meta.num_examples, MPI_INT, 0, MPI_VOTE_CACHE_TAG, MPI_COMM_WORLD);
                free(in_bag);
                // Get current oob_error and best_train_class so we can create the next bite
                //printf("Rank %d is receiving error and class info now\n", myrank);
                MPI_Recv(&oob_error, 1, MPI_DOUBLE, 0, MPI_VOTE_CACHE_TAG, MPI_COMM_WORLD, &status);
                MPI_Recv(best_train_class, train_data[0].meta.num_examples, MPI_INT, 0, MPI_VOTE_CACHE_TAG,
                         MPI_COMM_WORLD, &status);
            } else if (keep_going == AT_RECEIVE_NEW_DATA) {
                //printf("Proc %d for the signal to receive new data\n", myrank);
                receive_subset(&train_data[0], myrank, args);
            } else {
                //printf("Proc %d got the signal to cease and desist\n", myrank);
            }
            
            free_DT_Node(Tree, Books.next_unused_node);
        }
    }
}

void init_ensemble_file(CV_Metadata meta, CV_Partition part, int part_num, int fold_num, Args_Opts args) {
    if (args.save_trees) {
        args.datafile = av_strdup(part.partition_datafile[part_num]);
        args.base_filestem = av_strdup(part.partition_base_filestem[part_num]);
        args.data_path = av_strdup(part.partition_data_path[part_num]);
        set_output_filenames(&args, FALSE, TRUE);
        write_tree_file_header(args.num_trees, meta, fold_num, args.trees_file, args);
    }
}

void init_ensemble(DT_Ensemble *ensemble, CV_Metadata meta, Args_Opts args) {
    int i;
    if (args.num_trees > 0)
        ensemble->num_trees = args.num_trees;
    else if (args.auto_stop == TRUE)
        ensemble->num_trees = 100; // Start with 100 for stopping algorithm
    ensemble->num_classes = meta.num_classes;
    ensemble->num_attributes = meta.num_attributes;
    ensemble->attribute_types = meta.attribute_types;
    ensemble->num_training_examples = meta.num_examples;
    free(ensemble->num_training_examples_per_class);
    ensemble->num_training_examples_per_class = (int *)malloc(meta.num_classes * sizeof(int));
    for (i = 0; i < meta.num_classes; i++) {
        ensemble->num_training_examples_per_class[i] = meta.num_examples_per_class[i];
    }
    ensemble->Missing = (union data_point_union *)malloc(ensemble->num_attributes * sizeof(union data_point_union));
    for (i = 0; i < ensemble->num_attributes; i++) {
        if (ensemble->attribute_types[i] == DISCRETE)
            ensemble->Missing[i].Discrete = meta.Missing[i].Discrete;
        else if (ensemble->attribute_types[i] == CONTINUOUS && args.format != EXODUS_FORMAT)
            ensemble->Missing[i].Continuous = meta.Missing[i].Continuous;
    }
    ensemble->Trees = (DT_Node **)malloc(ensemble->num_trees * sizeof(DT_Node *));
    ensemble->Books = (Tree_Bookkeeping *)malloc(ensemble->num_trees * sizeof(Tree_Bookkeeping));
    ensemble->weights = (float *)malloc(ensemble->num_trees * sizeof(float));
}

