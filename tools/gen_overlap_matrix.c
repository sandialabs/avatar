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
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "array.h"
#include "regions.h"

int createFormats( int timesteps, int region_number, int label_number, int region_size, int mesh_number,
                   char *f_timestep, char *f_region_size_space, char *f_region_size,
                   char *f_overlap_val, char *f_column_percent, char *f_double_percent );
void print_comments(char *filename1, char *filename2, float threshold);
int labels_been_seen(int label, char *brand);
int labels_been_classed(int label, char *brand);
void extract_labels_been_classed(char *list, char *brand);
void _parse_range(char *str, int *num, int **range);
void add_set_to_list(int label, timestep ts, char **list);

int main(int argc, char** argv) {
    int i, j;
    int tr_region, cr_region;
    int tt, ct;
    int rc;
    
    char *truth_filename;
    char *comp_filename;
    FILE *truth_file = NULL;
    FILE *comp_file = NULL;
    
    timestep *truth_timesteps;
    region_file_metadata truth_metadata;
    int num_truth_timesteps;
    timestep *comp_timesteps;
    region_file_metadata comp_metadata;
    int num_comp_timesteps;
    region_max_values max_vals;
    
    char f_timestep[128], f_region_size_space[128];
    char f_region_size[128], f_column_percent[128];
    char f_overlap_val[128], f_double_percent[128];
    
    int **tr_sizes, **cr_sizes;
    int **sum_comp_regions;
    int **sum_truth_regions;
    int **over_seg_size;
    int **under_seg_size;
    char ***over_seg_list;
    char ***over_seg_plus_list;
    char ***under_seg_list;
    char ***under_seg_plus_list;
    int seg_list_realloc;
    int **over_seg_number;
    int **under_seg_number;
    
    int ***overlap;
    
    float threshold;
    
    if (argc < 3) {
        usage:
        printf("usage: %s [options] truth_region_file comparison_region_file\n", argv[0]);
        printf("options: \n");
        printf("   -t threshold : the segmentation threshold for determining correct-,\n");
        printf("                  under-, and over-segmentation.\n");
        printf("   -h           : print this help message\n");
        printf("\n");
        printf("Prints to stdout the overlap matrix for the two region files.\n");
        printf("\n");
        exit(-1);
    }
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t")) {
            threshold = atof(argv[i+1]);
            i++;
        } else if (!strncmp(argv[i], "-h", 2) || !strncmp(argv[i], "-", 1))
            goto usage;
        else {
            if (i+1 >= argc)
                goto usage;
            truth_filename = argv[i];
            comp_filename = argv[i+1];
            i+=1;
        }
    }
    
    truth_file = fopen(truth_filename, "r");
    if (truth_file == NULL) {
        fprintf(stderr, "Error: Unable to read truth file '%s'\n", truth_filename);
        exit(8);
    }
    comp_file = fopen(comp_filename, "r");
    if (comp_file == NULL) {
        fprintf(stderr, "Error: Unable to read comparison file '%s'\n", comp_filename);
        exit(8);
    }
    
    // Read region data
    rc = read_region_data(truth_file, &truth_timesteps, &truth_metadata);
    if (rc != 0) {
        fprintf(stderr, "Failed to open truth regions file: '%s'\n", truth_filename);
        exit(-8);
    }
    rc = read_region_data(comp_file, &comp_timesteps, &comp_metadata);
    if (rc != 0) {
        fprintf(stderr, "Failed to open test regions file: '%s'\n", comp_filename);
        exit(-8);
    }
    fclose(truth_file);
    fclose(comp_file);

    num_truth_timesteps = truth_metadata.num_timesteps;
    num_comp_timesteps = comp_metadata.num_timesteps;
    max_vals = init_max_vals();
    
    // Find max number of regions, max number of labels, max size, max number of nodes for formatting output
    get_region_data_max(truth_timesteps, truth_metadata, &max_vals);
    get_region_data_max(comp_timesteps, comp_metadata, &max_vals);
    
    int col_width = createFormats((num_truth_timesteps > num_comp_timesteps ? num_truth_timesteps : num_comp_timesteps),
                                  max_vals.number, max_vals.label, max_vals.size, max_vals.number,
                                  f_timestep, f_region_size_space, f_region_size, f_overlap_val,
                                  f_column_percent, f_double_percent );
    
//    printf("FORMATS:\nf_timestep = '%s'\nf_region_size_space = '%s'\nf_region_size = '%s'\nf_overlap_val = '%s'\nf_column_percent = '%s'\nf_double_percent = '%s'\n",
//           f_timestep, f_region_size_space, f_region_size, f_overlap_val, f_column_percent, f_double_percent);
    print_comments(truth_filename, comp_filename, threshold);
    
    // Generate overlap matrices for each timestep
    int max_timesteps = ( num_truth_timesteps > num_comp_timesteps ? num_truth_timesteps : num_comp_timesteps );
    overlap = (int***)malloc( max_timesteps * sizeof(int**) );
    
    tr_sizes = (int**)malloc(max_timesteps * sizeof(int*));
    cr_sizes = (int**)malloc(max_timesteps * sizeof(int*));
    
    sum_comp_regions = (int**)malloc(max_timesteps * sizeof(int*));
    sum_truth_regions = (int**)malloc(max_timesteps * sizeof(int*));
    over_seg_size = (int**)malloc(max_timesteps * sizeof(int*));
    under_seg_size = (int**)malloc(max_timesteps * sizeof(int*));
    over_seg_list = (char***)malloc(max_timesteps * sizeof(char**));
    over_seg_plus_list = (char***)malloc(max_timesteps * sizeof(char**));
    under_seg_list = (char***)malloc(max_timesteps * sizeof(char**));
    under_seg_plus_list = (char***)malloc(max_timesteps * sizeof(char**));
    // Overestimate the size of the list of regions as the number of digits in the largest region number
    // plus 1 (to account for the ',' separator) times the number of regions.
    seg_list_realloc = ( ( (int)log10(max_vals.number) + 1 ) + 1 ) * max_vals.number;
    over_seg_number = (int**)malloc(max_timesteps * sizeof(int*));
    under_seg_number = (int**)malloc(max_timesteps * sizeof(int*));

    int count = 0;
    tt = 0;
    ct = 0;
    while (tt < num_truth_timesteps && ct < num_comp_timesteps) {
        if (truth_timesteps[tt].timestep == comp_timesteps[ct].timestep) {
            
            // Reset sums
            sum_comp_regions[count] = calloc(max_vals.number, sizeof(int));
            sum_truth_regions[count] = calloc(max_vals.number, sizeof(int));

            // Compare regions for this timestep
            int num_tr = truth_timesteps[tt].num_regions;
            region *tr = truth_timesteps[tt].regions;
            int num_cr = comp_timesteps[ct].num_regions;
            region *cr = comp_timesteps[ct].regions;
            int *Intersection;

            overlap[count] = (int**)malloc( num_tr * sizeof(int*) );
            tr_sizes[count] = (int*)calloc(num_tr, sizeof(int));
            cr_sizes[count] = (int*)calloc(num_cr, sizeof(int));
            labels_been_seen(-1, "truth");
            for (tr_region = 0; tr_region < num_tr; tr_region++) {
                i = tr[tr_region].label;
                tr_sizes[count][i] += tr[tr_region].size;
                if (! labels_been_seen(i, "truth"))
                    overlap[count][i] = (int*)calloc( num_cr, sizeof(int) );
                for (cr_region = 0; cr_region < num_cr; cr_region++) {
                    j = cr[cr_region].label;
                    if (tr_region == 0)
                        cr_sizes[count][j] += cr[cr_region].size;
                    int n = array_intersection(tr[tr_region].members, tr[tr_region].size,
                                               cr[cr_region].members, cr[cr_region].size,
                                               &Intersection);
                    overlap[count][i][j] += n;
                    //printf("T=%d,TR=%d,TL=%d,CR=%d,CL=%d,O=%2d,TO=%2d\n", truth_timesteps[tt].timestep,
                    //                                                      tr_region,i,cr_region,j,n,
                    //                                                      overlap[count][i][j]);
                    sum_comp_regions[count][i] += overlap[count][i][j];
                    sum_truth_regions[count][j] += overlap[count][i][j];
                    free(Intersection);
                }
            }

            tt++;
            ct++;
            count++;

        } else if (comp_timesteps[ct].timestep < truth_timesteps[tt].timestep) {
            // Handle timestep for the comparison data that is not in the truth data
            ct++;
        } else if (truth_timesteps[tt].timestep < comp_timesteps[ct].timestep) {
            // Handle timestep for the truth data that is not in the comparison data
            tt++;
        }
    }
    // Handle extra truth timesteps
    for (i = tt; i < num_truth_timesteps; i++) {
    }
    // Handle extra comparison timesteps
    for (i = ct; i < num_comp_timesteps; i++) {
    }

    // Print Summary information
    
    count = 0;
    tt = 0;
    ct = 0;
    while (tt < num_truth_timesteps && ct < num_comp_timesteps) {
        if (truth_timesteps[tt].timestep == comp_timesteps[ct].timestep) {

            over_seg_size[count] = calloc(max_vals.number, sizeof(int));
            under_seg_size[count] = calloc(max_vals.number, sizeof(int));
            over_seg_list[count] = malloc(max_vals.number * sizeof(char*));
            over_seg_plus_list[count] = malloc(max_vals.number * sizeof(char*));
            under_seg_list[count] = malloc(max_vals.number * sizeof(char*));
            under_seg_plus_list[count] = malloc(max_vals.number * sizeof(char*));
            over_seg_number[count] = calloc(max_vals.number, sizeof(int));
            under_seg_number[count] = calloc(max_vals.number, sizeof(int));
            for (i = 0; i < max_vals.number; i++) {
                under_seg_plus_list[count][i] = NULL;
                over_seg_plus_list[count][i] = NULL;
            }

            int num_tr = truth_timesteps[tt].num_regions;
            region *tr = truth_timesteps[tt].regions;
            int num_cr = comp_timesteps[ct].num_regions;
            region *cr = comp_timesteps[ct].regions;

            labels_been_classed(-1, "truth");
            labels_been_classed(-1, "comp");
            
            labels_been_seen(-1, "truth");
            for (tr_region = 0; tr_region < num_tr; tr_region++) {
                i = tr[tr_region].label;
                if (labels_been_seen(i, "truth"))
                    continue;
                labels_been_seen(-1, "comp");
                for (cr_region = 0; cr_region < num_cr; cr_region++) {
                    j = cr[cr_region].label;
                    if (labels_been_seen(j, "comp"))
                        continue;
 
                    float t_fraction = (float)overlap[count][i][j] / (float)tr_sizes[count][i];
                    float c_fraction = (float)overlap[count][i][j] / (float)cr_sizes[count][j];
                    
                    if (t_fraction >= threshold && c_fraction < threshold) {
                        under_seg_size[count][j] += overlap[count][i][j];
                        under_seg_number[count][j]++;
                        if (under_seg_number[count][j] == 1) {
                            under_seg_list[count][j] = malloc(seg_list_realloc * sizeof(char));
                            sprintf(under_seg_list[count][j], "%d", i);
                        } else {
                            sprintf(under_seg_list[count][j], "%s,%d", under_seg_list[count][j], i);
                        }
                        add_set_to_list(i, truth_timesteps[tt], &under_seg_plus_list[count][j]);
                    } else if (t_fraction >= threshold && c_fraction >= threshold) {
                        under_seg_size[count][j] += overlap[count][i][j];
                        under_seg_number[count][j]++;
                        if (under_seg_number[count][j] == 1) {
                            under_seg_list[count][j] = malloc(seg_list_realloc * sizeof(char));
                            sprintf(under_seg_list[count][j], "%d", i);
                        } else {
                            sprintf(under_seg_list[count][j], "%s,%d", under_seg_list[count][j], i);
                        }
                        add_set_to_list(i, truth_timesteps[tt], &under_seg_plus_list[count][j]);
                        over_seg_size[count][i] += overlap[count][i][j];
                        over_seg_number[count][i]++;
                        if (over_seg_number[count][i] == 1) {
                            over_seg_list[count][i] = malloc(seg_list_realloc * sizeof(char));
                            sprintf(over_seg_list[count][i], "%d", j);
                        } else {
                            sprintf(over_seg_list[count][i], "%s,%d", over_seg_list[count][i], j);
                        }
                        add_set_to_list(j, comp_timesteps[ct], &over_seg_plus_list[count][i]);
                        labels_been_classed(i, "truth");
                        labels_been_classed(j, "comp");
                        printf("correct timestep=%d region_labels=%d/%d ",
                               truth_timesteps[tt].timestep, tr[tr_region].label, cr[cr_region].label);
                        printf("regions_numbers_meshes=");
                        char *cor_truth = NULL;
                        char *cor_test = NULL;
                        add_set_to_list(i, truth_timesteps[tt], &cor_truth);
                        add_set_to_list(j, comp_timesteps[ct], &cor_test);
                        printf("%s/%s\n", cor_truth, cor_test);
                        free(cor_truth);
                        free(cor_test);
                    } else if (t_fraction < threshold && c_fraction >= threshold) {
                        over_seg_size[count][i] += overlap[count][i][j];
                        over_seg_number[count][i]++;
                        if (over_seg_number[count][i] == 1) {
                            over_seg_list[count][i] = malloc(seg_list_realloc * sizeof(char));
                            sprintf(over_seg_list[count][i], "%d", j);
                        } else {
                            sprintf(over_seg_list[count][i], "%s,%d", over_seg_list[count][i], j);
                        }
                        add_set_to_list(j, comp_timesteps[ct], &over_seg_plus_list[count][i]);
                    }
                }
            }
            
            labels_been_seen(-1, "truth");
            for (tr_region = 0; tr_region < num_tr; tr_region++) {
                i = tr[tr_region].label;
                if (labels_been_seen(i, "truth"))
                    continue;

                float over_fraction = (float)over_seg_size[count][i] / (float)tr_sizes[count][i];
                if (over_seg_number[count][i] > 1 && over_fraction >= threshold) {
                    char *i_list = NULL;
                    add_set_to_list(i, truth_timesteps[tt], &i_list);
                    labels_been_classed(i, "truth");
                    extract_labels_been_classed(over_seg_list[truth_timesteps[tt].timestep][i], "comp");
                    printf("over-seg timestep=%d region_labels=%d/%s regions_numbers_meshes=%s/%s\n",
                           truth_timesteps[tt].timestep, i, over_seg_list[truth_timesteps[tt].timestep][i],
                           i_list, over_seg_plus_list[truth_timesteps[tt].timestep][i]);
                    free(i_list);
                }
            }
            labels_been_seen(-1, "comp");
            for (cr_region = 0; cr_region < num_cr; cr_region++) {
                j = cr[cr_region].label;
                if (labels_been_seen(j, "comp"))
                    continue;
                
                float under_fraction = (float)under_seg_size[count][j] / (float)cr_sizes[count][j];
                if (under_seg_number[count][j] > 1 && under_fraction >= threshold) {
                    char *j_list = NULL;
                    add_set_to_list(j, comp_timesteps[ct], &j_list);
                    labels_been_classed(j, "comp");
                    extract_labels_been_classed(under_seg_list[truth_timesteps[tt].timestep][j], "truth");
                    printf("under-seg timestep=%d region_labels=%s/%d regions_numbers_meshes=%s/%s\n",
                           truth_timesteps[tt].timestep, under_seg_list[truth_timesteps[tt].timestep][j], j,
                           under_seg_plus_list[truth_timesteps[tt].timestep][j], j_list);
                    free(j_list);
                }
            }
            for (tr_region = 0; tr_region < num_tr; tr_region++) {
                i = tr[tr_region].label;
                if (! labels_been_classed(i, "truth")) {
                    printf("missed timestep=%d region=%d ", truth_timesteps[tt].timestep, i);
                    char *missed = NULL;
                    add_set_to_list(i, truth_timesteps[tt], &missed);
                    printf("regions_numbers_meshes=%s\n", missed);
                    free(missed);
                }
            }
            for (cr_region = 0; cr_region < num_cr; cr_region++) {
                j = cr[cr_region].label;
                if (! labels_been_classed(j, "comp")) {
                    printf("noise timestep=%d region=%d ", truth_timesteps[tt].timestep, j);
                    char *noise = NULL;
                    add_set_to_list(j, comp_timesteps[ct], &noise);
                    printf("regions_numbers_meshes=%s\n", noise);
                    free(noise);
                }
            }
            
            tt++;
            ct++;
            count++;

        } else if (comp_timesteps[ct].timestep < truth_timesteps[tt].timestep) {
            // Handle timestep for the comparison data that is not in the truth data
            ct++;
        } else if (truth_timesteps[tt].timestep < comp_timesteps[ct].timestep) {
            // Handle timestep for the truth data that is not in the comparison data
            tt++;
        }
    }
    // Handle extra truth timesteps
    for (i = tt; i < num_truth_timesteps; i++) {
    }
    // Handle extra comparison timesteps
    for (i = ct; i < num_comp_timesteps; i++) {
    }
    
    printf("\n");
    printf("# GORY DETAILS:\n\n");
    
    count = 0;
    tt = 0;
    ct = 0;
    while (tt < num_truth_timesteps && ct < num_comp_timesteps) {
        if (truth_timesteps[tt].timestep == comp_timesteps[ct].timestep) {

            labels_been_classed(-1, "truth");
            labels_been_classed(-1, "comp");

            // Reset sums
            for (i = 0; i < max_vals.number; i++) {
                sum_comp_regions[count][i] = 0;
                sum_truth_regions[count][i] = 0;
            }
            // Compare regions for this timestep
            int num_tr = truth_timesteps[tt].num_regions;
            region *tr = truth_timesteps[tt].regions;
            int num_cr = comp_timesteps[ct].num_regions;
            region *cr = comp_timesteps[ct].regions;

            printf(f_timestep, truth_timesteps[tt].timestep);

            // Print raw numbers
            if (num_tr > 0 && num_cr > 0) {
                printf(f_region_size_space, " ");
                labels_been_seen(-1, "comp");
                for (cr_region = 0; cr_region < num_cr; cr_region++) {
                    j = cr[cr_region].label;
                    if (labels_been_seen(j, "comp"))
                        continue;
                    printf(f_region_size, j, cr_sizes[count][j]);
                }
                printf("\n");
                labels_been_seen(-1, "truth");
                for (tr_region = 0; tr_region < num_tr; tr_region++) {
                    i = tr[tr_region].label;
                    if (labels_been_seen(i, "truth"))
                        continue;
                    printf(f_region_size, i, tr_sizes[count][i]);
                    labels_been_seen(-1, "comp");
                    for (cr_region = 0; cr_region < num_cr; cr_region++) {
                        j = cr[cr_region].label;
                        if (labels_been_seen(j, "comp"))
                            continue;
                        printf(f_overlap_val, overlap[count][i][j]);
                    }
                    printf("\n");
                }
                if (num_tr > 0)
                    printf("\n\n");
            }
            
            // Print correct detection numbers
            if (num_tr > 0 && num_cr > 0) {
                printf(f_region_size_space, " ");
                labels_been_seen(-1, "comp");
                for (cr_region = 0; cr_region < num_cr; cr_region++) {
                    j = cr[cr_region].label;
                    if (labels_been_seen(j, "comp"))
                        continue;
                    printf(f_region_size, j, cr_sizes[count][j]);
                }
                printf("\n");
                labels_been_seen(-1, "truth");
                for (tr_region = 0; tr_region < num_tr; tr_region++) {
                    i = tr[tr_region].label;
                    if (labels_been_seen(i, "truth"))
                        continue;
                    printf(f_region_size, i, tr_sizes[count][i]);
                    labels_been_seen(-1, "comp");
                    for (cr_region = 0; cr_region < num_cr; cr_region++) {
                        j = cr[cr_region].label;
                        if (labels_been_seen(j, "comp"))
                            continue;
                        if (overlap[count][i][j] == 0)
                            printf(f_region_size_space, "-  ");
                        else {
                            float t_fraction = (float)overlap[count][i][j] / (float)tr_sizes[count][i];
                            float c_fraction = (float)overlap[count][i][j] / (float)cr_sizes[count][j];
                            if (t_fraction >= threshold && c_fraction >= threshold) {
                                labels_been_classed(i, "truth");
                                labels_been_classed(j, "comp");
                                printf(f_double_percent, "C<", t_fraction * 100.0, "^", c_fraction * 100.0);
                            } else {
                                printf(f_double_percent, "<", t_fraction * 100.0, "^", c_fraction * 100.0);
                            }
                        }
                    }
                    printf("\n");
                }
                printf("\n\n");
            }

            // Print over- and under-segmentation numbers
            if (num_tr > 0 && num_cr > 0) {
                printf(f_region_size_space, " ");
                labels_been_seen(-1, "comp");
                for (cr_region = 0; cr_region < num_cr; cr_region++) {
                    j = cr[cr_region].label;
                    if (labels_been_seen(j, "comp"))
                        continue;
                    printf(f_region_size, j, cr_sizes[count][j]);
                }
                printf("\n");
                labels_been_seen(-1, "truth");
                for (tr_region = 0; tr_region < num_tr; tr_region++) {
                    i = tr[tr_region].label;
                    if (labels_been_seen(i, "truth")) {
                        continue;
                    }
                    printf(f_region_size, i, tr_sizes[count][i]);
                    labels_been_seen(-1, "comp");
                    for (cr_region = 0; cr_region < num_cr; cr_region++) {
                        j = cr[cr_region].label;
                        if (labels_been_seen(j, "comp"))
                            continue;
                        if (overlap[count][i][j] == 0)
                            printf(f_region_size_space, "-  ");
                        else {
                            float t_fraction = (float)overlap[count][i][j] / (float)tr_sizes[count][i];
                            float c_fraction = (float)overlap[count][i][j] / (float)cr_sizes[count][j];
                            
                            char designator[4];
                            strcpy(designator, "");
                            if (c_fraction >= threshold && over_seg_number[count][i] > 1) {
                                strcat(designator, "O");
                                labels_been_classed(i, "truth");
                                labels_been_classed(j, "comp");
                            }
                            if (t_fraction >= threshold && under_seg_number[count][j] > 1) {
                                strcat(designator, "U");
                                labels_been_classed(i, "truth");
                                labels_been_classed(j, "comp");
                            }
                            strcat(designator, "<");
                            printf(f_double_percent, designator, t_fraction * 100.0, "^", c_fraction * 100.0);
                        }
                    }
                    float over_fraction = (float)over_seg_size[count][i] / (float)tr_sizes[count][i];
                    if (! labels_been_classed(i, "truth"))
                        printf("  M %3.0f%%\n", over_fraction * 100.0);
                    else if (over_seg_number[count][i] > 1 && over_fraction >= threshold)
                        printf("  O %3.0f%%\n", over_fraction * 100.0);
                    else
                        printf("    %3.0f%%\n", over_fraction * 100.0);
                }
                printf(f_region_size_space, " ");
                labels_been_seen(-1, "comp");
                for (cr_region = 0; cr_region < num_cr; cr_region++) {
                    j = cr[cr_region].label;
                    if (labels_been_seen(j, "comp"))
                        continue;
                    float under_fraction = (float)under_seg_size[count][j] / (float)cr_sizes[count][j];
                    if (! labels_been_classed(j, "comp"))
                        printf(f_column_percent, " N", under_fraction * 100.0);
                    else if (under_seg_number[count][j] > 1 && under_fraction >= threshold)
                        printf(f_column_percent, " U", under_fraction * 100.0);
                    else
                        printf(f_column_percent, "", under_fraction * 100.0);
                }
                printf("\n\n");
            }
            
            for (i = 0; i < get_num_labels(truth_timesteps[tt]); i++) {
                free(overlap[count][i]);
            }
            free(overlap[count]);
            
            tt++;
            ct++;
            count++;
            
        } else if (comp_timesteps[ct].timestep < truth_timesteps[tt].timestep) {
            // Handle timestep for the comparison data that is not in the truth data
            ct++;
        } else if (truth_timesteps[tt].timestep < comp_timesteps[ct].timestep) {
            // Handle timestep for the truth data that is not in the comparison data
            tt++;
        }
    }
    // Handle extra truth timesteps
    for (i = tt; i < num_truth_timesteps; i++) {
    }
    // Handle extra comparison timesteps
    for (i = ct; i < num_comp_timesteps; i++) {
    }
    
    // clean up
    labels_been_seen(-1, "truth");
    labels_been_seen(-1, "comp");
    
    free_region_data(truth_timesteps, truth_metadata);
    free_region_data(comp_timesteps,  comp_metadata);
    
    for (i = 0; i < max_timesteps; i++) {
        free(sum_truth_regions[i]);
        free(sum_comp_regions[i]);
        free(over_seg_size[i]);
        free(under_seg_size[i]);
        for (j = 0; j < max_vals.number; j++) {
            if (over_seg_number[i][j] > 0) {
                free(over_seg_list[i][j]);
                free(over_seg_plus_list[i][j]);
            }
            if (under_seg_number[i][j] > 0) {
                free(under_seg_list[i][j]);
                free(under_seg_plus_list[i][j]);
            }
        }
        free(over_seg_list[i]);
        free(over_seg_plus_list[i]);
        free(under_seg_list[i]);
        free(under_seg_plus_list[i]);
        free(over_seg_number[i]);
        free(under_seg_number[i]);
        free(tr_sizes[i]);
        free(cr_sizes[i]);
    }
    free(sum_truth_regions);
    free(sum_comp_regions);
    free(over_seg_size);
    free(under_seg_size);
    free(over_seg_number);
    free(under_seg_number);
    free(over_seg_list);
    free(over_seg_plus_list);
    free(under_seg_list);
    free(under_seg_plus_list);
    free(tr_sizes);
    free(cr_sizes);

    free(overlap);
    
    exit(0);
}

int labels_been_classed(int label, char *brand) {
    int i;
    int classedit = 0;
    
    static int *classed_truth_labels;
    static int *classed_comp_labels;
    static int num_classed_truth_labels = 0;
    static int num_classed_comp_labels = 0;
    
    // Initialize
    if (label == -1) {
        if (! strcmp(brand, "truth") && num_classed_truth_labels > 0) {
            free(classed_truth_labels);
            num_classed_truth_labels = 0;
        } else if (! strcmp(brand, "comp") && num_classed_comp_labels > 0) {
            free(classed_comp_labels);
            num_classed_comp_labels = 0;
        }
        return -1;
    }
    
    if (! strcmp(brand, "truth")) {
        for (i = 0; i < num_classed_truth_labels; i++) {
            if (label == classed_truth_labels[i])
                classedit = 1;
        }
        if (! classedit) {
            num_classed_truth_labels++;
            if (num_classed_truth_labels == 1)
                classed_truth_labels = (int *)malloc(sizeof(int));
            else
                classed_truth_labels = (int *)realloc(classed_truth_labels, num_classed_truth_labels * sizeof(int));
            classed_truth_labels[num_classed_truth_labels - 1] = label;
        }
    } else if (! strcmp(brand, "comp")) {
        for (i = 0; i < num_classed_comp_labels; i++) {
            if (label == classed_comp_labels[i])
                classedit = 1;
        }
        if (! classedit) {
            num_classed_comp_labels++;
            if (num_classed_comp_labels == 1)
                classed_comp_labels = (int *)malloc(sizeof(int));
            else
                classed_comp_labels = (int *)realloc(classed_comp_labels, num_classed_comp_labels * sizeof(int));
            classed_comp_labels[num_classed_comp_labels - 1] = label;
        }
    }
    
    return classedit;
}

void extract_labels_been_classed(char *list, char *brand) {
    int *labels;
    int i, num_labels;
    _parse_range(list, &num_labels, &labels);
    for (i = 0; i < num_labels; i++)
        labels_been_classed(labels[i], brand);
}

void _parse_range(char *str, int *num, int **range) {
    
    char *token, *ptr_d, *ptr_c;
    int start, end;
    int i;
    int c = 0;
    int cur_alloc = 10;
    *range = (int *)malloc(cur_alloc * sizeof(int));
    *num = 0;

    token = strdup(str);
    // Handle each string separated by ',' -- but this skips the last one ...
    while ((ptr_c = strchr(token, ','))) {
        *(ptr_c) = '\0';
        if ((ptr_d = strchr(token, '-'))) {
            *(ptr_d) = '\0';
            start = atoi(token);
            token = ++ptr_d;
            end = atoi(token);
            *num += end - start + 1;
        } else {
            start = atoi(token);
            end = start;
            (*num)++;
        }
        
        if (*num > cur_alloc) {
            cur_alloc = *num + 10;
            *range = (int *)realloc(*range, cur_alloc * sizeof(int));
        }
        for (i = start; i <= end; i++) {
            (*range)[c++] = i;
        }
        
        token = strdup(++ptr_c);
    }

    // ... handle the last one
    if ((ptr_d = strchr(token, '-'))) {
        *(ptr_d) = '\0';
        start = atoi(token);
        token = ++ptr_d;
        end = atoi(token);
        *num += end - start + 1;
    } else {
        start = atoi(token);
        end = start;
        (*num)++;
    }

    if (*num > cur_alloc) {
        cur_alloc = *num + 10;
        *range = (int *)realloc(*range, cur_alloc * sizeof(int));
    }
    for (i = start; i <= end; i++) {
        (*range)[c++] = i;
    }

}

int labels_been_seen( int label, char *brand ) {
    int i;
    int seenit = 0;
    
    static int *seen_truth_labels;
    static int *seen_comp_labels;
    static int num_seen_truth_labels = 0;
    static int num_seen_comp_labels = 0;
    
    // Initialize
    if (label == -1) {
        if (! strcmp(brand, "truth") && num_seen_truth_labels > 0) {
            free(seen_truth_labels);
            num_seen_truth_labels = 0;
        } else if (! strcmp(brand, "comp") && num_seen_comp_labels > 0) {
            free(seen_comp_labels);
            num_seen_comp_labels = 0;
        }
        return -1;
    }
    
    if (! strcmp(brand, "truth")) {
        for (i = 0; i < num_seen_truth_labels; i++) {
            if (label == seen_truth_labels[i])
                seenit = 1;
        }
        if (! seenit) {
            num_seen_truth_labels++;
            if (num_seen_truth_labels == 1)
                seen_truth_labels = (int *)malloc(sizeof(int));
            else
                seen_truth_labels = (int *)realloc(seen_truth_labels, num_seen_truth_labels * sizeof(int));
            seen_truth_labels[num_seen_truth_labels - 1] = label;
        }
    } else if (! strcmp(brand, "comp")) {
        for (i = 0; i < num_seen_comp_labels; i++) {
            if (label == seen_comp_labels[i])
                seenit = 1;
        }
        if (! seenit) {
            num_seen_comp_labels++;
            if (num_seen_comp_labels == 1)
                seen_comp_labels = (int *)malloc(sizeof(int));
            else
                seen_comp_labels = (int *)realloc(seen_comp_labels, num_seen_comp_labels * sizeof(int));
            seen_comp_labels[num_seen_comp_labels - 1] = label;
        }
    }
    
    return seenit;
}

void add_set_to_list(int label, timestep ts, char **list) {
    int i;
    char *char_list;
    int *int_list;
    
    //printf("\n\nFinding region numbers for label %d and adding to '%s'\n\n", label, *list);
    
    for (i = 0; i < get_num_meshes(ts); i++) {
        if (label_to_number_list(label, i, ts, &char_list, &int_list) > 0) {
            int partial_size = strlen(char_list) + ( i > 0 ? (int)log10(i) : 1 ) + 1 + 3;
            if (*list == NULL || strlen(*list) == 0) {
                (*list) = (char *)malloc(partial_size * sizeof(char));
                sprintf(*list, "(%s)%d", char_list, i);
            } else {
                (*list) = (char *)realloc(*list, (partial_size + strlen(*list) + 1) * sizeof(char));
                sprintf(*list, "%s,(%s)%d", *list, char_list, i);
            }
            free(char_list);
        }
        free(int_list);
    }
}


int createFormats( int timesteps, int region_number, int region_label, int region_size, int mesh_number,
                   char *f_timestep, char *f_region_size_space, char *f_region_size,
                   char *f_overlap_val, char *f_column_percent, char *f_double_percent) {
    
    int n1, n2, n3, n4, n5;
    
    // The number of digits needed to display the maximum timestep
    n1 = (int)log10(timesteps) + 1;
    sprintf(f_timestep, "Timestep = %%%dd\n\n", n1);
    
    // The number of digits needed to display the maximum region number, label number, and size
    n1 = (int)log10(region_number) + 1;
    n2 = (int)log10(region_label) + 1;
    n3 = (int)log10(region_size) + 1;
    n4 = (int)log10(mesh_number) + 1;
    n5 = n2 + n3;
    if (n5 < 7) {
        n2 += 7 - n5;
        n5 = n2 + n3;
    }
    
    int dpf = (int)ceil((n5 + 3)/2.0) - 2;
    if (dpf < 3) {
        n1 += 3 - dpf;
        dpf = 3;
    }

    sprintf(f_region_size_space, " %%%ds", n5 + 4);
    sprintf(f_region_size, "%%%dd(%%%dd) ", n2 + 2, n3);
    sprintf(f_column_percent, "%%2s %%%d.0f%%%% ", n5);
    sprintf(f_overlap_val, "%%%dd  ", n5 + 3);
    sprintf(f_double_percent, "%%%ds%%3.0f%%1s%%3.0f%%%% ", dpf);
    
    return (n5 + 3);
}

void print_comments(char *filename1, char *filename2, float threshold) {
    printf("# OVERLAP MATRIX\n");
    printf("#\n");
    printf("# Comparing '%s' (truth) and '%s' (test)\n", filename1, filename2);
    printf("# Using a threshold of %f\n", threshold);
    printf("#\n");
    printf("# The SUMMARY section contains a terse description of the segmentation results\n");
    printf("#\n");
    printf("# Each line contains the following four, space-delimited elements:\n");
    printf("#     1. Segmentation Type\n");
    printf("#            as one of correct, over-seg, under-seg, missed, noise\n");
    printf("#     2. Timestep\n");
    printf("#     3. Region Labels\n");
    printf("#            as truth1[,truth2...]/test1[,test2...]\n");
    printf("#     4. Region Numbers and Mesh Number corresponding to each Region Label\n");
    printf("#            as (number1[,number2...])mesh\n");
    printf("# The Region Numbers are the original, unjoined, unique region designator.\n");
    printf("# The Region Labels are non-unique and are the same for all connected component regions.\n");
    printf("#\n");
    printf("# For each timestep the GORY DETAILS is comprised of 3 tables:\n");
    printf("#\n");
    printf("# The first table shows RAW OVERLAP numbers\n");
    printf("#     Table row/column headings = I(J)\n");
    printf("#         where: I = region label\n");
    printf("#                J = size of region\n");
    printf("#         and truth regions are row headers; test regions are column headers\n");
    printf("#     Table cell entries = number of overlap points\n");
    printf("#\n");
    printf("# The second table shows CORRECT DETECTION based on the threshold\n");
    printf("#     Table row/column headings = I(J) as above\n");
    printf("#     Table cell entries = [C]<Ptr^Pte\n");
    printf("#         where: C    = indicates that this pair of regions satisifes both\n");
    printf("#                       metrics\n");
    printf("#                <Ptr = percent of points in the truth region that are in the\n");
    printf("#                       overlap set\n");
    printf("#                ^Pte = percent of points in the test region that are in the\n");
    printf("#                       overlap set\n");
    printf("#\n");
    printf("# The third table shows OVER- and UNDER-SEGMENTATION based on the threshold,\n");
    printf("# MISSED regions and NOISE regions\n");
    printf("#     Table row/column headings = I(J) as above\n");
    printf("#     Table cell entries = [OU]<Ptr^Pte\n");
    printf("#         where: O    = indicates that Ptr >= threshold\n");
    printf("#                U    = indicates that Pte >= threshold\n");
    printf("#                <Ptr = percent of points in the truth region that are in the\n");
    printf("#                       overlap set\n");
    printf("#                ^Pte = percent of points in the test region that are in the\n");
    printf("#                       overlap set\n");
    printf("#     Percentages in last column = total percentage overlap of all test regions\n");
    printf("#                                  with truth region possibly prepended with:\n");
    printf("#                                      O = total overlap / truth region size >= threshold\n");
    printf("#                                      M = truth regions was missed\n");
    printf("#     Percentages in last row    = total percentage overlap of all truth regions\n");
    printf("#                                  with test region possibly prepended with:\n");
    printf("#                                      U = total overlap / test region size >= threshold\n");
    printf("#                                      N = test region is noise\n");
    printf("\n");
    printf("# SUMMARY:\n");
    printf("\n");
}
