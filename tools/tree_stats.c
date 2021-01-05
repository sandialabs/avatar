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
#ifndef _GNU_SOURCE
  #include "getopt.h"
#else
  #include <getopt.h>
#endif
#include "../src/version_info.h"
#include "../src/crossval.h"
#include "../src/attr_stats.h"
#include "../src/tree.h"
#include "../src/options.h"
#include "../src/rw_data.h"
#include "../src/memory.h"
#include "../src/reset.h"

struct Global_Args_t {
  char* modelfile;
  char* namesfile;
  char* datafile;
} MyArgs;

// Prototypes for helper functions.
void _display_usage(void);
void _process_opts(int argc, char** argv);

void
display_usage()
{
  _display_usage();
}

int
main(int argc, char** argv)
{
  Attr_Stats* stats = NULL;
  DT_Ensemble model;
  Args_Opts ens_opts;
  CV_Metadata meta;
  CV_Class classmeta;

  // Cosmin: reset metadata & model
  //printf("before reset meta\n"); fflush(stdout);
  reset_CV_Metadata(&meta);
  reset_DT_Ensemble(&model);

  // parse arguments and options
  _process_opts(argc, argv);
  ens_opts = process_opts(0, NULL);  // all this line does is default init.
  ens_opts.trees_file = MyArgs.modelfile;
  ens_opts.names_file = MyArgs.namesfile;
  
  // read names file
  //printf("before read names\n"); fflush(stdout);
  if (!read_names_file(&meta, &classmeta, &ens_opts, TRUE)) {
    fprintf(stderr, "Error reading names file %s\n", ens_opts.names_file);
    exit(-1);
  }
  //printf("after read names\n"); fflush(stdout);

  // load model into memory
  read_ensemble(&model, -1, 0, &ens_opts);
  //printf("after read ensemble\n"); fflush(stdout);

  // compute tree-based statistics of feature importance
  stats = malloc_attr_stats(&meta, ens_opts.num_skipped_features, ens_opts.skipped_features);
  compute_feature_imp(&model, stats);

  // print statistics
  write_feature_imp(stdout, stats);

  // clean up
  free_attr_stats(stats);
  stats = NULL;
  free_CV_Class(classmeta);
  free_DT_Ensemble(model, TEST_MODE);
  // REVIEW-2011-01-06: Currently no easy way to free CV_Metadata.
  // Fixing this likely requires refactoring code in memory.c under free_CV_Subset or free_CV_Dataset.

  return 0;
}

void
_display_usage()
{
  printf("usage: %s [options] modelfile namesfile\n\n", "tree_stats");
  printf("Computes tree-based statistics of feature importance.  Importance scores\n"
	 "are computed from the structure of the trees in the ensemble and how much\n"
	 "each feature contributes to classifying a set of data points.  The\n"
	 "statistics are printed to STDOUT with the format:\n"
	 "\n"
	 "    # Feature Mass Mass_No_Dup Path Deviance Depth\n"
	 "      x1      0.06    0.04     0.05  0.10     0.3\n"
	 "      x2      0.10    0.06     0.09  0.05     0.01\n"
	 "     ...\n"
	 "\n"
	 "STATISTICS\n\n"
	 "Brief descriptions of the statistics follow.  See [1] for full details. All\n"
	 "statistics computed per tree, summed over all trees, and then normalized to\n"
	 "sum to 1.  According to [1], all Mass-based stats are similar and are good\n"
	 "proxies for Breiman's sensitivity analysis.  Depth is included for\n"
	 "completeness but does not correlate well with the ranking from sensitivity\n"
	 "analysis; it should generally not be used.  Deviance was developed after [1]\n"
	 "was published, and as a result it is unknown whether it is preferable to\n"
	 "Path (for example).  One point in favor of Deviance is supporting literature\n"
	 "in the statistics community about various properties.\n"
	 "\n"
	 "* Mass: The volume/mass of data points partially classified by the feature.\n"
	 "  Count the number of data points in each leaf.  These counts are the data\n"
	 "  masses for each leaf.  The mass for each internal node is the sum of the\n"
	 "  masses of its children.  A feature's mass statistic is the sum of masses\n"
	 "  over the nodes that test the feature to split the data.\n\n"
	 "* Mass_No_Dup: Removes possible duplicate counting in Mass caused by\n"
	 "  testing a continuous feature multiple times in a tree.  If feature X\n"
	 "  appears multiple times on the path from a leaf to the root, the leaf's\n"
	 "  mass is only counted once towards X's importance.\n\n"
	 "* Path: A more nuanced mass-based statistic.  A leaf's mass is evenly\n"
	 "  apportioned to every feature test on the path from the leaf to the tree\n"
	 "  root.  Thus, if feature X is tested multiple times, it gets more credit\n"
	 "  for helping to categorize the points in the leaf.\n\n"
	 "* Deviance: The change in deviance between parent node and child nodes,\n"
	 "  summed over all nodes in the tree that test the feature.  The deviance\n"
	 "  at a node is defined as:\n"
	 "      deviance = -2 * [ln Pr(labels | node=negative) \n"
	 "                      +ln Pr(labels | node=positive)]\n"
	 "               = -2 * [#negative * ln(#negative/#total)\n"
	 "                      +#positive * ln(#positive/#total)]\n"
	 "  This is very closely related to entropy of the class labels in the set\n"
	 "  of examples that reach the node.\n\n"
	 "* Depth: Importance based on the feature's depth in the tree, with higher\n"
	 "  importance implied for features with lower depth (i.e., closer to tree\n"
	 "  root).  More precisely, sum \n"
	 "       weight = 1 / (depth(node) + 1)\n"
	 "  over all nodes that test feature X to get the depth importance of feature\n"
	 "  X.  (Recall that the depth of the tree root is 0.) \n"
	 "\n"
	 "OPTIONS\n\n"
	 "  --datafile f   : Base statistics on data in f, instead of from train set.\n"
	 "  -h             : show this help message\n"
	 "\n"
	 "REFERENCES\n\n"
	 " [1] R. Caruana, M. Elhawary, D. Fink, W.M. Hochachka, S. Kelling, A. Munson,\n"
	 "     M. Riedewald, & D. Sorokina.  Mining citizen science data to predict\n"
	 "     prevalence of wild bird species.  In Proc. of Intl Conf on Knowledge\n"
	 "     Discovery and Data Mining, 2006.\n"
	 );
}

static const char* opt_string = "+hd:";

static const struct option long_opts[] = {
  {"datafile", required_argument, NULL, 'd'},
  {"help", no_argument, NULL, 'h'},
  {NULL, no_argument, NULL, 0}
};

void
_process_opts(int argc, char** argv)
{
  int opt = 0;
  int long_index = 0;
  Boolean found_opt = 0;

  // Initialize
  MyArgs.modelfile = NULL;
  MyArgs.namesfile = NULL;
  MyArgs.datafile = NULL;

  // Grab options.
  found_opt = -1 != (opt = getopt_long(argc, argv, opt_string, long_opts, &long_index));
  while (found_opt) {
    switch (opt) {
      case 'd':
	MyArgs.datafile = optarg;
	break;
      case 'h':
	_display_usage();
	break;
      default:
	break;
    }
    found_opt = -1 != (opt = getopt_long(argc, argv, opt_string, long_opts, &long_index));
  }

  // Grab required arguments.
  argc -= optind;
  if (argc < 2) {
    fprintf(stderr, "Missing model and/or names file arguments.\n");
    _display_usage();
    exit(0);
  }

  argv += optind;
  MyArgs.modelfile = argv[0];
  MyArgs.namesfile = argv[1];
}
