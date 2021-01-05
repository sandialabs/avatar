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
#include <mpi.h>
#include <string.h>
#include <math.h>
#include "crossval.h"
#include "mpiL.h"

MPI_Datatype MPI_OPTIONS;
MPI_Datatype MPI_EXAMPLE;
MPI_Datatype MPI_TREENODE;

void derive_MPI_TREENODE() {
    int i;
    int num = 6;
    int length[num];
    MPI_Aint disp[num];
    MPI_Datatype type[num];
    MPI_Aint baseaddress;
    DT_Node Node;

    MPI_Address(&Node, &baseaddress);

    for (i = 0; i < num; i++) {
        length[i] = 1;
        type[i] = MPI_INT;
    }
    type[4] = MPI_FLOAT;
    
    MPI_Address(&Node.branch_type,      &disp[0]);
    MPI_Address(&Node.attribute,        &disp[1]);
    MPI_Address(&Node.num_branches,     &disp[2]);
    MPI_Address(&Node.num_errors,       &disp[3]);
    MPI_Address(&Node.branch_threshold, &disp[4]);
    MPI_Address(&Node.attribute_type,   &disp[5]);
    
    for (i = 0; i < num; i++)
        disp[i] -= baseaddress;
    
    MPI_Type_struct(num, length, disp, type, &MPI_TREENODE);
    MPI_Type_commit(&MPI_TREENODE);
}

void derive_MPI_EXAMPLE() {
    int i;
    int num = 10;
    int length[num];
    MPI_Aint disp[num];
    MPI_Datatype type[num];
    MPI_Aint baseaddress;
    CV_Example Example;
    
    MPI_Address(&Example, &baseaddress);
    
    for (i = 0; i < num; i++) {
        length[i] = 1;
        type[i] = MPI_INT;
    }
    
    MPI_Address(&Example.global_id_num,        &disp[0]);
    MPI_Address(&Example.random_gid,           &disp[1]);
    MPI_Address(&Example.fclib_seq_num,        &disp[2]);
    MPI_Address(&Example.fclib_id_num,         &disp[3]);
    MPI_Address(&Example.containing_class_num, &disp[4]);
    MPI_Address(&Example.predicted_class_num,  &disp[5]);
    MPI_Address(&Example.containing_fold_num,  &disp[6]);
    MPI_Address(&Example.bl_clump_num,         &disp[7]);
    MPI_Address(&Example.is_missing,           &disp[8]);
    MPI_Address(&Example.in_bag,               &disp[9]);
    
    for (i = 0; i < num; i++)
        disp[i] -= baseaddress;
    
    MPI_Type_struct(num, length, disp, type, &MPI_EXAMPLE);
    MPI_Type_commit(&MPI_EXAMPLE);
}

void derive_MPI_OPTIONS() {
    
    int i, j;
    int num = 100;
    int length[num];
    MPI_Aint disp[num];
    MPI_Datatype type[num];
    MPI_Aint baseaddress;
    Args_Opts Args;
    
    MPI_Address(&Args, &baseaddress);
    
    for (j = 0; j < num; j++) {
        length[j] = 1;
        type[j] = MPI_INT;
    }
    
    i = 0;
    //MPI_Address(&Args.do_ts_based,                  &disp[i++]);
    MPI_Address(&Args.go4it,                        &disp[i++]);
    MPI_Address(&Args.do_5x2_cv,                    &disp[i++]);
    MPI_Address(&Args.do_nfold_cv,                  &disp[i++]);
    MPI_Address(&Args.do_rigorous_strat,            &disp[i++]);
    MPI_Address(&Args.num_folds,                    &disp[i++]);
    MPI_Address(&Args.num_train_times,              &disp[i++]);
    MPI_Address(&Args.num_test_times,               &disp[i++]);
    MPI_Address(&Args.write_folds,                  &disp[i++]);
    type[i] = MPI_LONG;
    MPI_Address(&Args.random_seed,                  &disp[i++]);
    MPI_Address(&Args.format,                       &disp[i++]);
    MPI_Address(&Args.verbosity,                    &disp[i++]);
    MPI_Address(&Args.caller,                       &disp[i++]);
    MPI_Address(&Args.do_training,                  &disp[i++]);
    MPI_Address(&Args.do_testing,                   &disp[i++]);
    MPI_Address(&Args.truth_column,                 &disp[i++]);
    MPI_Address(&Args.num_skipped_features,         &disp[i++]);
    MPI_Address(&Args.num_trees,                    &disp[i++]);
    MPI_Address(&Args.auto_stop,                    &disp[i++]);
    MPI_Address(&Args.slide_size,                   &disp[i++]);
    MPI_Address(&Args.build_size,                   &disp[i++]);
    MPI_Address(&Args.split_on_zero_gain,           &disp[i++]);
    MPI_Address(&Args.dynamic_bounds,               &disp[i++]);
    MPI_Address(&Args.minimum_examples,             &disp[i++]);
    MPI_Address(&Args.split_method,                 &disp[i++]);
    MPI_Address(&Args.save_trees,                   &disp[i++]);
    MPI_Address(&Args.random_forests,               &disp[i++]);
    MPI_Address(&Args.random_attributes,            &disp[i++]);
    type[i] = MPI_FLOAT;
    MPI_Address(&Args.random_subspaces,             &disp[i++]);
    MPI_Address(&Args.collapse_subtree,             &disp[i++]);
    type[i] = MPI_FLOAT;
    MPI_Address(&Args.do_bagging,                   &disp[i++]);
    MPI_Address(&Args.bag_size,                     &disp[i++]);
    MPI_Address(&Args.majority_bagging,             &disp[i++]);
    MPI_Address(&Args.do_ivote,                     &disp[i++]);
    MPI_Address(&Args.bite_size,                    &disp[i++]);
    type[i] = MPI_FLOAT;
    MPI_Address(&Args.ivote_p_factor,               &disp[i++]);
    MPI_Address(&Args.majority_ivoting,             &disp[i++]);
    MPI_Address(&Args.do_smote,                     &disp[i++]);
    MPI_Address(&Args.smote_knn,                    &disp[i++]);
    MPI_Address(&Args.smote_Ln,                     &disp[i++]);
    MPI_Address(&Args.smote_type,                   &disp[i++]);
    MPI_Address(&Args.do_boosting,                  &disp[i++]);
    MPI_Address(&Args.do_smoteboost,                &disp[i++]);
    MPI_Address(&Args.do_balanced_learning,         &disp[i++]);
    MPI_Address(&Args.do_noising,                   &disp[i++]);
    MPI_Address(&Args.output_predictions,           &disp[i++]);
    MPI_Address(&Args.output_probabilities,         &disp[i++]);
    MPI_Address(&Args.output_confusion_matrix,      &disp[i++]);
    MPI_Address(&Args.debug,                        &disp[i++]);
    MPI_Address(&Args.read_folds,                   &disp[i++]);
    MPI_Address(&Args.use_opendt_shuffle,           &disp[i++]);
    MPI_Address(&Args.run_regression_test,          &disp[i++]);
    MPI_Address(&Args.break_ties_randomly,          &disp[i++]);
    MPI_Address(&Args.common_mpi_rand48_seed,       &disp[i++]);
    num = i;
    
    for (j = 0; j < num; j++)
        disp[j] -= baseaddress;
    
    MPI_Type_struct(num, length, disp, type, &MPI_OPTIONS);
    MPI_Type_commit(&MPI_OPTIONS);
}

void broadcast_options(Args_Opts *Args) {
    MPI_Bcast(Args, 1, MPI_OPTIONS, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
}

/*
 * Not currently used
 *
void _broadcast_subset(CV_Subset *sub, int myrank, Args_Opts args) {
    int i;
    MPI_Datatype MPI_DISTINCT_VALUES;

    MPI_Bcast(&sub->meta.num_classes, 1, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&sub->meta.num_attributes, 1, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&sub->meta.num_fclib_seq, 1, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&sub->meta.num_examples, 1, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&sub->malloc_examples, 1, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (myrank != 0) {
        if (args.format == EXODUS_FORMAT)
            sub->meta.global_offset = (int *)malloc((sub->meta.num_fclib_seq + 1) * sizeof(int));
        sub->meta.attribute_types = (Attribute_Type *)malloc(sub->meta.num_attributes * sizeof(Attribute_Type));
        sub->discrete_used = (Boolean *)malloc(sub->meta.num_attributes * sizeof(Boolean));
        sub->meta.num_discrete_values = (int *)malloc(sub->meta.num_attributes * sizeof(int));
        sub->high = (int *)malloc(sub->meta.num_attributes * sizeof(int));
        sub->low = (int *)malloc(sub->meta.num_attributes * sizeof(int));
        sub->examples = (CV_Example *)malloc(sub->meta.num_examples * sizeof(CV_Example));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (args.format == EXODUS_FORMAT)
        MPI_Bcast(sub->meta.global_offset, sub->meta.num_fclib_seq + 1, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(sub->meta.attribute_types, sub->meta.num_attributes, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    if (args.format != EXODUS_FORMAT) {
        MPI_Bcast(sub->discrete_used, sub->meta.num_attributes, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
        MPI_Bcast(sub->meta.num_discrete_values, sub->meta.num_attributes, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    }
    MPI_Bcast(sub->high, sub->meta.num_attributes, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(sub->low, sub->meta.num_attributes, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myrank != 0) {
        sub->float_data = (float **)malloc(sub->meta.num_attributes * sizeof(float *));
        for (i = 0; i < sub->meta.num_attributes; i++)
            sub->float_data[i] = (float *)malloc((sub->high[i] + 1) * sizeof(float));
    }
    for (i = 0; i < sub->meta.num_attributes; i++)
        MPI_Bcast(sub->float_data[i], sub->high[i] + 1, MPI_FLOAT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);

    MPI_Type_contiguous(sub->meta.num_attributes, MPI_INT, &MPI_DISTINCT_VALUES);
    MPI_Type_commit(&MPI_DISTINCT_VALUES);

    int number_sent = 0;
    int number_to_send;
    int position = 0;
    int packsize;
    while(number_sent < sub->meta.num_examples) {
        int c = (int)(AT_MPI_BUFMAX / (sizeof(int) * (sub->meta.num_attributes + 7))) - (sub->meta.num_examples - number_sent);
        if (c > 0)
            number_to_send = sub->meta.num_examples - number_sent;
        else
            number_to_send = (int)(AT_MPI_BUFMAX / (sizeof(int)*(sub->meta.num_attributes+7)));

        int p1, p2;
        MPI_Pack_size(number_to_send, MPI_EXAMPLE, MPI_COMM_WORLD, &p1);
        MPI_Pack_size(number_to_send, MPI_DISTINCT_VALUES, MPI_COMM_WORLD, &p2);
        packsize = p1 + p2;
        char* buff;
        buff = (char *)malloc(packsize * sizeof(char));

        if (myrank == 0) {
            for (i = number_sent; i < number_sent + number_to_send; i++) {
                MPI_Pack(&sub->examples[i], 1, MPI_EXAMPLE, buff, packsize, &position, MPI_COMM_WORLD);
                MPI_Pack(sub->examples[i].distinct_attribute_values, 1, MPI_DISTINCT_VALUES, buff,
                         packsize, &position, MPI_COMM_WORLD);
            }
        }

        MPI_Bcast(buff, packsize, MPI_PACKED, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);

        position=0;
        if (myrank != 0) {
            for (i = number_sent; i < number_sent + number_to_send; i++) {
                MPI_Unpack(buff, packsize, &position, &sub->examples[i], 1, MPI_EXAMPLE, MPI_COMM_WORLD);
                sub->examples[i].distinct_attribute_values = (int *)malloc(sub->meta.num_attributes * sizeof(int));
                MPI_Unpack(buff, packsize, &position, sub->examples[i].distinct_attribute_values,
                           1, MPI_DISTINCT_VALUES, MPI_COMM_WORLD);
            }
        }

        free(buff);
        number_sent += number_to_send;
    }

}
 *
 */

// Rank 0 sends the CV_Subset sub to rank send_to

void send_subset(CV_Subset *sub, int send_to, Args_Opts args) {
    int i;
    int size1, size2, packsize;
    int position;
    int mpires;
    
    int num_int_to_pack = 0;
    int num_float_to_pack = 0;

    // Pack num_classes, num_attributes, num_fclib_seq, num_examples, malloc_examples
    num_int_to_pack += 5;
    // Pack num_examples_per_class;
    num_int_to_pack += sub->meta.num_classes;
    // Pack global_offset
    if (args.format == EXODUS_FORMAT)
        num_int_to_pack += sub->meta.num_fclib_seq + 1;
    // Pack attribute_types
    num_int_to_pack += sub->meta.num_attributes;
    // Pack discrete_used and num_discrete_values
    if (args.format != EXODUS_FORMAT)
        num_int_to_pack += 2 * sub->meta.num_attributes;
    // Pack high and low
    num_int_to_pack += 2 * sub->meta.num_attributes;

    // Pack float_data;
    for (i = 0; i < sub->meta.num_attributes; i++)
        if (sub->meta.attribute_types[i] == CONTINUOUS)
            num_float_to_pack += sub->high[i] + 1;
    
    //num_int_to_pack = 2;
    //num_float_to_pack = 0;
    //printf("Packing %d ints and %d floats\n", num_int_to_pack, num_float_to_pack);
    MPI_Pack_size(num_int_to_pack, MPI_INT, MPI_COMM_WORLD, &size1);
    MPI_Pack_size(num_float_to_pack, MPI_FLOAT, MPI_COMM_WORLD, &size2);
    if (size1 + size2 > AT_MPI_BUFMAX) {
        fprintf(stderr, "Increase AT_MPI_BUFMAX to hold %d\n", size1+size2);
        exit(-8);
    }
    
    char *buff;
    packsize = size1 + size2;
    buff = (char *)malloc(packsize * sizeof(char));
    position = 0;
    
    //printf("Rank 0 packing num_classes = %d\n", nc);
    MPI_Pack(&sub->meta.num_classes, 1, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    //printf("Rank 0 res for packing num_classes = %d and new position is %d\n", mpires, position);
    //printf("Rank 0 packing num_attributes = %d\n", sub->meta.num_attributes);
    MPI_Pack(&sub->meta.num_attributes, 1, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    //printf("Rank 0 res for packing num_atts = %d and new position is %d\n", mpires, position);
    MPI_Pack(&sub->meta.num_fclib_seq, 1, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    MPI_Pack(&sub->meta.num_examples, 1, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    MPI_Pack(sub->meta.num_examples_per_class, sub->meta.num_classes, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    MPI_Pack(&sub->malloc_examples, 1, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    
    if (args.format == EXODUS_FORMAT)
        MPI_Pack(sub->meta.global_offset, sub->meta.num_fclib_seq + 1, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    MPI_Pack(sub->meta.attribute_types, sub->meta.num_attributes, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    if (args.format != EXODUS_FORMAT) {
        MPI_Pack(sub->discrete_used, sub->meta.num_attributes, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
        MPI_Pack(sub->meta.num_discrete_values, sub->meta.num_attributes, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    }
    MPI_Pack(sub->high, sub->meta.num_attributes, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    MPI_Pack(sub->low, sub->meta.num_attributes, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
    
    for (i = 0; i < sub->meta.num_attributes; i++)
        if (sub->meta.attribute_types[i] == CONTINUOUS)
            MPI_Pack(sub->float_data[i], sub->high[i] + 1, MPI_FLOAT, buff, packsize, &position, MPI_COMM_WORLD);
    
    mpires = MPI_Send(buff, packsize, MPI_PACKED, send_to, MPI_SUBSETMETA_TAG, MPI_COMM_WORLD);
    if (! check_mpi_error(mpires, "MPI_Send MPI_SUBSETMETA_TAG")) exit(-8);
    free(buff);
    
    //MPI_Datatype MPI_DISTINCT_VALUES;
    //MPI_Type_contiguous(sub->meta.num_attributes, MPI_INT, &MPI_DISTINCT_VALUES);
    //MPI_Type_commit(&MPI_DISTINCT_VALUES);
    
    int number_sent = 0;
    int number_to_send;
    position = 0;
    
    //printf("Rank 0 is sending data to %d...\n", send_to);
    while(number_sent < sub->meta.num_examples) {
        int c = (int)((AT_MPI_BUFMAX - sizeof(int)) /
                                      (sizeof(int) * (sub->meta.num_attributes + 7))) - (sub->meta.num_examples - number_sent);
        if (c > 0)
            number_to_send = sub->meta.num_examples - number_sent;
        else
            number_to_send = (int)((AT_MPI_BUFMAX - sizeof(int)) / (sizeof(int)*(sub->meta.num_attributes+7)));

        MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &packsize);
        MPI_Pack_size(number_to_send, MPI_EXAMPLE, MPI_COMM_WORLD, &size1);
        MPI_Pack_size(number_to_send * sub->meta.num_attributes, MPI_INT, MPI_COMM_WORLD, &size2);
        packsize += size1 + size2;
        buff = (char *)malloc(packsize * sizeof(char));
        //printf("Rank 0 is sending %d samples in a message of length %d\n", number_to_send, packsize);
        mpires = MPI_Pack(&number_to_send, 1, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
        if (! check_mpi_error(mpires, "MPI_Pack")) exit(-8);
        for (i = number_sent; i < number_sent + number_to_send; i++) {
            mpires = MPI_Pack(&sub->examples[i], 1, MPI_EXAMPLE, buff, packsize, &position, MPI_COMM_WORLD);
            if (! check_mpi_error(mpires, "MPI_Pack examples")) exit(-8);
            //MPI_Pack(sub->examples[i].distinct_attribute_values, 1, MPI_DISTINCT_VALUES, buff,
            //         packsize, &position, MPI_COMM_WORLD);
            mpires = MPI_Pack(sub->examples[i].distinct_attribute_values, sub->meta.num_attributes, MPI_INT, buff,
                              packsize, &position, MPI_COMM_WORLD);
            if (! check_mpi_error(mpires, "MPI_Pack disticnt_attribute_values")) exit(-8);
        }
        
        mpires = MPI_Send(buff, packsize, MPI_PACKED, send_to, MPI_SUBSETDATA_TAG, MPI_COMM_WORLD);
        if (! check_mpi_error(mpires, "MPI_Send MPI_SUBSETDATA_TAG")) exit(-8);
        //printf("Rank 0 just sent %d data to %d\n", number_to_send, send_to);
        free(buff);
        number_sent += number_to_send;
    }
}

void receive_subset(CV_Subset *sub, int myrank, Args_Opts args) {
    int i;
    int position;
    char *buff;
    int buff_length;
    MPI_Status status;
    
    //printf("Rank %d waiting to receive meta\n", myrank);
    MPI_Probe(AT_MPI_ROOT_RANK, MPI_SUBSETMETA_TAG, MPI_COMM_WORLD, &status);
    //printf("Rank %d sees a waiting message\n", myrank);
    MPI_Get_count(&status, MPI_CHAR, &buff_length);
    buff = (char *)malloc(buff_length * sizeof(char));
    //printf("Rank %d sees a waiting message of length %d\n", myrank, len);
    MPI_Recv(buff, buff_length, MPI_PACKED, AT_MPI_ROOT_RANK, MPI_SUBSETMETA_TAG, MPI_COMM_WORLD, &status);
    //printf("Rank %d MPI_Recv error = %d\n", myrank, status.MPI_ERROR);
    position = 0;
    
    MPI_Unpack(buff, buff_length, &position, &sub->meta.num_classes, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buff, buff_length, &position, &sub->meta.num_attributes, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buff, buff_length, &position, &sub->meta.num_fclib_seq, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buff, buff_length, &position, &sub->meta.num_examples, 1, MPI_INT, MPI_COMM_WORLD);
    sub->meta.num_examples_per_class = (int *)malloc(sub->meta.num_classes * sizeof(int));
    MPI_Unpack(buff, buff_length, &position, sub->meta.num_examples_per_class, sub->meta.num_classes, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buff, buff_length, &position, &sub->malloc_examples, 1, MPI_INT, MPI_COMM_WORLD);
    
    if (args.format == EXODUS_FORMAT)
        sub->meta.global_offset = (int *)malloc((sub->meta.num_fclib_seq + 1) * sizeof(int));
    sub->meta.attribute_types = (Attribute_Type *)malloc(sub->meta.num_attributes * sizeof(Attribute_Type));
    sub->discrete_used = (Boolean *)malloc(sub->meta.num_attributes * sizeof(Boolean));
    sub->meta.num_discrete_values = (int *)malloc(sub->meta.num_attributes * sizeof(int));
    sub->high = (int *)malloc(sub->meta.num_attributes * sizeof(int));
    sub->low = (int *)malloc(sub->meta.num_attributes * sizeof(int));
    sub->examples = (CV_Example *)malloc(sub->meta.num_examples * sizeof(CV_Example));
    
    if (args.format == EXODUS_FORMAT)
        MPI_Unpack(buff, buff_length, &position, sub->meta.global_offset, sub->meta.num_fclib_seq + 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buff, buff_length, &position, sub->meta.attribute_types, sub->meta.num_attributes, MPI_INT, MPI_COMM_WORLD);
    if (args.format != EXODUS_FORMAT) {
        MPI_Unpack(buff, buff_length, &position, sub->discrete_used, sub->meta.num_attributes, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, buff_length, &position, sub->meta.num_discrete_values, sub->meta.num_attributes, MPI_INT, MPI_COMM_WORLD);
    }
    MPI_Unpack(buff, buff_length, &position, sub->high, sub->meta.num_attributes, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buff, buff_length, &position, sub->low, sub->meta.num_attributes, MPI_INT, MPI_COMM_WORLD);
    
    sub->float_data = (float **)malloc(sub->meta.num_attributes * sizeof(float *));
    for (i = 0; i < sub->meta.num_attributes; i++)
        sub->float_data[i] = (float *)malloc((sub->high[i] + 1) * sizeof(float));
    
    for (i = 0; i < sub->meta.num_attributes; i++)
        if (sub->meta.attribute_types[i] == CONTINUOUS)
            MPI_Unpack(buff, buff_length, &position, sub->float_data[i], sub->high[i] + 1, MPI_FLOAT, MPI_COMM_WORLD);
    
    //MPI_Datatype MPI_DISTINCT_VALUES;
    //MPI_Type_contiguous(sub->meta.num_attributes, MPI_INT, &MPI_DISTINCT_VALUES);
    //MPI_Type_commit(&MPI_DISTINCT_VALUES);
    free(buff);
    
    //printf("Rank %d going into receive mode for data\n", myrank);
    int number_to_receive = -1;
    int number_received = 0;
    while (number_received < sub->meta.num_examples) {
        //printf("Rank %d has received %d data\n", myrank, number_received);
        //printf("Rank %d waiting for message\n", myrank);
        MPI_Probe(AT_MPI_ROOT_RANK, MPI_SUBSETDATA_TAG, MPI_COMM_WORLD, &status);
        //printf("Rank %d sees a message\n",myrank);
        MPI_Get_count(&status, MPI_CHAR, &buff_length);
        //printf("Rank %d sees a message of length %d\n", myrank, buff_length);
        buff = (char *)malloc(buff_length * sizeof(char));
        MPI_Recv(buff, buff_length, MPI_PACKED, AT_MPI_ROOT_RANK, MPI_SUBSETDATA_TAG, MPI_COMM_WORLD, &status);
        //printf("Rank %d MPI_Recv error = %d\n", myrank, status.MPI_ERROR);
        
        position=0;
        MPI_Unpack(buff, buff_length, &position, &number_to_receive, 1, MPI_INT, MPI_COMM_WORLD);
        //printf("Rank %d is expecting %d data samples\n", myrank, number_to_receive);
        for (i = number_received; i < number_received + number_to_receive; i++) {
            MPI_Unpack(buff, buff_length, &position, &sub->examples[i], 1, MPI_EXAMPLE, MPI_COMM_WORLD);
            sub->examples[i].distinct_attribute_values = (int *)malloc(sub->meta.num_attributes * sizeof(int));
            //MPI_Unpack(buff, max_packsize, &position, sub->examples[i].distinct_attribute_values,
            //           1, MPI_DISTINCT_VALUES, MPI_COMM_WORLD);
            MPI_Unpack(buff, buff_length, &position, sub->examples[i].distinct_attribute_values, sub->meta.num_attributes,
                       MPI_INT, MPI_COMM_WORLD);
        }
        
        number_received += number_to_receive;
        free(buff);
    }
}

void receive_one_tree(char *buf, int *pos, DT_Node **tree, int num_classes, int nodes) {
    int j;
    int packsize = AT_MPI_BUFMAX;
    
    (*tree) = (DT_Node *)malloc(nodes * sizeof(DT_Node));
    for (j = 0; j < nodes; j++) {
        //printf("Unpacking node %d\n", j);
        MPI_Unpack(buf, packsize, pos, &(*tree)[j], 1, MPI_TREENODE, MPI_COMM_WORLD);
        //printf("While unpacking tree, found node %d to be %d\n", j, (*tree)[j].branch_type);
        if ((*tree)[j].branch_type == LEAF) {
            //printf("Leaf...\n");
            MPI_Unpack(buf, packsize, pos, &(*tree)[j].Node_Value.class_label, 1, MPI_INT, MPI_COMM_WORLD);
            (*tree)[j].class_count = (int *)malloc(num_classes * sizeof(int));
            MPI_Unpack(buf, packsize, pos, (*tree)[j].class_count, num_classes, MPI_INT, MPI_COMM_WORLD);
            (*tree)[j].class_probs = (float *)malloc(num_classes * sizeof(float));
            MPI_Unpack(buf, packsize, pos, (*tree)[j].class_probs, num_classes, MPI_FLOAT, MPI_COMM_WORLD);

        } else if ((*tree)[j].branch_type == BRANCH) {
            //printf("Branch...\n");
            //printf("malloc Node_Value.branch to %d\n", (*tree)[j].num_branches);
            (*tree)[j].Node_Value.branch = (int *)malloc((*tree)[j].num_branches * sizeof(int));
            //printf("malloc done\n");
            MPI_Unpack(buf, packsize, pos, (*tree)[j].Node_Value.branch, (*tree)[j].num_branches,
                       MPI_INT, MPI_COMM_WORLD);
            //printf("unpack done\n");
        }
    }
}

void send_trees(DT_Node **trees, Tree_Bookkeeping *books, int num_classes, int my_num) {
    int i, j;
    int try_packsize;
    int packsize = 0;
    int size1, size2, size3, size4;
    int position;
    int num_to_send;
    int num_sent = 0;
    
    while (num_sent < my_num) {
        
        // pack the number of trees and the number of nodes in each tree
        MPI_Pack_size(my_num+1, MPI_INT, MPI_COMM_WORLD, &try_packsize);
        
        i = num_sent;
        while (i < my_num && try_packsize < AT_MPI_BUFMAX) {
            // pack the ith tree
            for (j = 0; j < books[i].next_unused_node; j++) {
                MPI_Pack_size(1, MPI_TREENODE, MPI_COMM_WORLD, &size1);
                MPI_Pack_size( (trees[i][j].branch_type == LEAF ? 1 : trees[i][j].num_branches),
                                MPI_INT, MPI_COMM_WORLD, &size2);
                try_packsize += size1 + size2;
                if (trees[i][j].branch_type == LEAF) {
                    MPI_Pack_size(num_classes, MPI_INT, MPI_COMM_WORLD, &size3);
                    MPI_Pack_size(num_classes, MPI_FLOAT, MPI_COMM_WORLD, &size4);
                    try_packsize += size3 + size4;
                }
            }
            
            if (try_packsize < AT_MPI_BUFMAX) {
                packsize = try_packsize;
                i++;
            }
        }
        
        num_to_send = i - num_sent;
        if (num_to_send == 0) {
            fprintf(stderr, "At least one tree is too big for the buffer\n");
            exit(-1);
        }
        //printf("Sending %d trees\n", num_to_send);
        
        char *buff;
        buff = (char *)malloc(packsize * sizeof(char));
        position = 0;
        
        MPI_Pack(&num_to_send, 1, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
        for (i = num_sent; i < num_sent + num_to_send; i++) {
            //printf("Tree %d has %d nodes\n", i, books[i].next_unused_node);
            MPI_Pack(&books[i].next_unused_node, 1, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
        }
        //printf("Start packing trees ...\n");
        for (i = num_sent; i < num_sent + num_to_send; i++) {
            //printf("Packing tree %d which has %d nodes\n", i, books[i].next_unused_node);
            for (j = 0; j < books[i].next_unused_node; j++) {
                MPI_Pack(&trees[i][j], 1, MPI_TREENODE, buff, packsize, &position, MPI_COMM_WORLD);
                //printf("While packing tree %d, found node %d to be %d\n", i, j, trees[i][j].branch_type);
                if (trees[i][j].branch_type == LEAF) {
                    MPI_Pack(&trees[i][j].Node_Value.class_label, 1, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
                    MPI_Pack(trees[i][j].class_count, num_classes, MPI_INT, buff, packsize, &position, MPI_COMM_WORLD);
                    MPI_Pack(trees[i][j].class_probs, num_classes, MPI_FLOAT, buff, packsize, &position, MPI_COMM_WORLD);
                } else if (trees[i][j].branch_type == BRANCH) {
    //                printf("Packing %d branches\n", trees[i][j].num_branches);
                    MPI_Pack(trees[i][j].Node_Value.branch, trees[i][j].num_branches, MPI_INT, buff, packsize,
                             &position, MPI_COMM_WORLD);
                }
            }
        }
        
        MPI_Ssend(buff, packsize, MPI_PACKED, 0, MPI_TREENODE_TAG, MPI_COMM_WORLD);
        num_sent += num_to_send;
        free(buff);
    }
}

int check_mpi_error(int err, char *label) {
    if (err == MPI_SUCCESS) {
        //printf("%s: MPI_SUCCESS\n", label);
        return 1;
    } else if (err == MPI_ERR_BUFFER) {
        printf("%s: MPI_ERR_BUFFER\n", label);
    } else if (err == MPI_ERR_COUNT) {
        printf("%s: MPI_ERR_COUNT\n", label);
    } else if (err == MPI_ERR_TYPE) {
        printf("%s: MPI_ERR_TYPE\n", label);
    } else if (err == MPI_ERR_TAG) {
        printf("%s: MPI_ERR_TAG\n", label);
    } else if (err == MPI_ERR_COMM) {
        printf("%s: MPI_ERR_COMM\n", label);
    } else if (err == MPI_ERR_RANK) {
        printf("%s: MPI_ERR_RANK\n", label);
    } else if (err == MPI_ERR_ROOT) {
        printf("%s: MPI_ERR_ROOT\n", label);
    } else if (err == MPI_ERR_TRUNCATE) {
        printf("%s: MPI_ERR_TRUNCATE\n", label);
    } else {
        printf("%s: (%d) UNKNOWN???\n", label, err);
    }
    return 0;
}
