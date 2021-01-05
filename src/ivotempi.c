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
#include "crossval.h"
#include "ivotempi.h"
#include "tree.h"
#include "util.h"
#include "evaluate.h"

void make_bite_mpi(CV_Subset *src, CV_Subset *bite, double oob_error, int *best_train_class, Args_Opts args) {
    int i, j, k;
    double selection_probability;
    
    if (args.bite_size <= 0 || args.bite_size > src->meta.num_examples) {
        fprintf(stderr, "ERROR: invalid number of examples in bite: %d\n", args.bite_size);
        exit(-8);
    }

    copy_subset_meta(*src, bite, args.bite_size);
    copy_subset_data(*src, bite);
    bite->meta.num_examples = args.bite_size;
    //printf("Current oob_error = %g\n", oob_error);
    selection_probability = oob_error / (1.0 - oob_error);
    // Re-initialize to false
    for (i = 0; i < src->meta.num_examples; i++)
        src->examples[i].in_bag = FALSE;
    
    i = 0;
    while (i < args.bite_size) {
        // Pick random sample with replacement
        j = lrand48() % src->meta.num_examples;
        if (best_train_class[j] == src->examples[j].containing_class_num) {
            // Got it right - include with probability c(k)=selection_probability
            if (drand48() < selection_probability) {
                copy_example_data(src->meta.num_attributes, src->examples[j], &(bite->examples[i]));
                src->examples[j].in_bag = TRUE;
                i++;
            }
        } else {
            // Got it wrong - include this sample
            copy_example_data(src->meta.num_attributes, src->examples[j], &(bite->examples[i]));
            src->examples[j].in_bag = TRUE;
            i++;
        }
    }
    if (args.debug) {
        for (k = 0; k < bite->meta.num_examples; k++)
            printf("Data for Bag:%d Att:0 = %10g\n",
                   k, bite->float_data[0][bite->examples[k].distinct_attribute_values[0]]);
    }
}
/*
double compute_oob_error_rate(DT_Node *tree, CV_Subset train_data, Vote_Cache *cache, Args_Opts args) {
    int i, j;
    int num_errors = 0;
    int num_oob_examples = 0;
    int num_correct_for_this_tree = 0;
    int this_class, current_winner;
    int this_class_votes, current_winner_votes;
    int num_ties;
    int tied_classes[cache->num_classes];
    
    // Classify all training data (entire training set -- not any single voting bite)
    // Only OOB examples are used to compute the oob error rate
    for (i = 0; i < cache->num_train_examples; i++) {
        // If this is an OOB example ...
        if (train_data.examples[i].in_bag == FALSE) {
            
            this_class = classify_example(tree, train_data.examples[i], train_data.float_data);
            
            // Update number of correct classifications for this tree for average accuracy
            if (this_class == train_data.examples[i].containing_class_num)
                num_correct_for_this_tree++;

            // Compute voted accuracy for all trees
            current_winner = cache->best_train_class[i];
            // Update class votes
            cache->oob_class_votes[i][this_class]++;
            this_class_votes = cache->oob_class_votes[i][this_class];
            if (current_winner > -1)
                current_winner_votes = cache->oob_class_votes[i][current_winner];
            else
                current_winner_votes = 0;
            
            // Compare the new number of votes for this_class with the current winner
            if (current_winner < 0) {
                // First time voting for this example
                cache->best_train_class[i] = this_class;
            } else if (this_class_votes > current_winner_votes) {
                cache->best_train_class[i] = this_class;
            } else if (this_class_votes == current_winner_votes) {
                num_ties = 0;
                for (j = 0; j < cache->num_classes; j++)
                    if (cache->oob_class_votes[i][j] == cache->oob_class_votes[i][this_class])
                        tied_classes[num_ties++] = j;
                if (num_ties == 1 || args.break_ties_randomly == FALSE)
                    cache->best_train_class[i] = tied_classes[0];
                else
                    cache->best_train_class[i] = tied_classes[lrand48() % num_ties];
            }
            
            // Update number of errors and number of oob examples
            num_oob_examples++;
            if (cache->best_train_class[i] != train_data.examples[i].containing_class_num)
                num_errors++;
            
        }
    }
    
    // Update average accuracy
    cache->average_train_accuracy = ( cache->average_train_accuracy * (float)(cache->current_classifier_count - 1) +
                                     ((float)num_correct_for_this_tree / (float)num_oob_examples) ) /
                                    (float)cache->current_classifier_count;
    
    return (double)num_errors/(double)num_oob_examples;
}
    
double compute_test_error_rate(DT_Node *tree, CV_Subset test_data, Vote_Cache *cache, Args_Opts args) {
    int i, j;
    int num_errors = 0;
    int this_class, current_winner;
    int this_class_votes, current_winner_votes;
    int num_ties;
    int tied_classes[cache->num_classes];
    
    // Classify all the testing data since we're tossing the trees.
    int num_correct_for_this_tree = 0;
    int num_examples_for_accuracy = cache->num_test_examples;
    for (i = 0; i < cache->num_test_examples; i++) {
        this_class = classify_example(tree, test_data.examples[i], test_data.float_data);
        
        // Compute number of errors for this tree for average accuracy
        if (this_class == test_data.examples[i].containing_class_num)
            num_correct_for_this_tree++;
        
        // Compute voted accuracy for all trees
        current_winner = cache->best_test_class[i];
        // Update class votes
        cache->class_votes_test[i][this_class]++;
        this_class_votes = cache->class_votes_test[i][this_class];
        if (current_winner > -1)
            current_winner_votes = cache->class_votes_test[i][current_winner];
        else
            current_winner_votes = 0;
        
        // Compare the new number of votes for this_class with the current winner
        if (current_winner < 0) {
            // First time voting for this example
            cache->best_test_class[i] = this_class;
            //printf("First time voting for example %d: set to class %d\n", i, this_class);
        } else if (this_class_votes > current_winner_votes) {
            cache->best_test_class[i] = this_class;
            //printf("For example %d current winning class %d has %d votes. New winning class %d has %d votes\n",
            //        i, current_winner, current_winner_votes, this_class, this_class_votes);
        } else if (this_class_votes == current_winner_votes) {
            // There is a tie between at least two classes.
            // Find all classes involved in the tie and randomly assign one as the best
            num_ties = 0;
            for (j = 0; j < cache->num_classes; j++)
                if (cache->class_votes_test[i][j] == cache->class_votes_test[i][this_class])
                    tied_classes[num_ties++] = j;
            if (num_ties == 1 || args.break_ties_randomly == FALSE)
                cache->best_test_class[i] = tied_classes[0];
            else
                cache->best_test_class[i] = tied_classes[lrand48() % num_ties];
        }

        // Update number of errors
        if (cache->best_test_class[i] != test_data.examples[i].containing_class_num)
            num_errors++;

    }
    
    // Update average accuracy
    cache->average_test_accuracy = ( cache->average_test_accuracy * (float)(cache->current_classifier_count - 1) +
                                     ((float)num_correct_for_this_tree / (float)num_examples_for_accuracy) ) /
                                     (float)cache->current_classifier_count;

    return (double)num_errors/(double)num_examples_for_accuracy;
}
*/
