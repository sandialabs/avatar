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
#include "array.h"
#include "regions.h"
#ifndef _GNU_SOURCE
  #include "getopt.h"
#else
  #include <getopt.h>
#endif

struct Global_Args_t {
    char *filename;
    int *timestep_list;
    int num_timesteps;
    int print_timesteps;
    int *label_list;
    int num_labels;
    int print_labels;
    int *mesh_list;
    int num_meshes;
    int print_meshes;
    int *number_list;
    int num_numbers;
    int print_numbers;
    int print_members;
    int print_sizes;
    int terse_output;
} Args;

void _parse_range(char *str, int *num, int **range);
void _process_opts(int argc, char **argv);
void _display_usage(char *prog);

int main(int argc, char** argv) {
    
    int rc, i, j, k;
    FILE *region_file;
    timestep *timesteps, ts;
    region_file_metadata metadata;
    region_max_values max_vals;
    
    _process_opts(argc, argv);
    
    if (! strcmp(Args.filename, "STDIN")) {
        region_file = stdin;
    } else {
        if ((region_file = fopen(Args.filename, "r")) == NULL) {
            fprintf(stderr, "Error: unable to read regions file: '%s'\n", Args.filename);
            exit(8);
        }
    }
    if ((rc = read_region_data(region_file, &timesteps, &metadata)) != 0) {
        fprintf(stderr, "Error: failed to parse regions file: '%s'\n", Args.filename);
        exit(7);
    }
    get_region_data_max(timesteps, metadata, &max_vals);

    for (i = 0; i < Args.num_timesteps; i++)
        if (Args.timestep_list[i] > metadata.num_timesteps)
            fprintf(stderr, "Warning: timestep %d was not found -- skipping\n", Args.timestep_list[i] );
    
    for (i = 0; i < metadata.num_timesteps; i++) {
        if (! find_int(timesteps[i].timestep, Args.num_timesteps, Args.timestep_list))
            continue;
        
        ts = timesteps[i];
        
        for (j = 0; j < ts.num_regions; j++) {
            if (
                 (Args.num_numbers == 0 || find_int(ts.regions[j].number, Args.num_numbers, Args.number_list)) &&
                 (Args.num_labels  == 0 || find_int(ts.regions[j].label,  Args.num_labels,  Args.label_list)) &&
                 (Args.num_meshes  == 0 || find_int(ts.regions[j].mesh,   Args.num_meshes,  Args.mesh_list))
               ) {
                
                if (Args.print_timesteps) {
                    if (Args.terse_output)
                        printf("%d:", ts.timestep);
                    else
                        printf("Timestep: %d\n", ts.timestep);
                }
                if (Args.print_numbers) {
                    if (Args.terse_output)
                        printf("%d:", ts.regions[j].number);
                    else
                        printf(" Region Number: %d\n", ts.regions[j].number);
                }
                if (Args.print_labels) {
                    if (Args.terse_output)
                        printf("%d:", ts.regions[j].label);
                    else
                        printf("  Region Label: %d\n", ts.regions[j].label);
                }
                if (Args.print_meshes) {
                    if (Args.terse_output)
                        printf("%d:", ts.regions[j].mesh);
                    else
                        printf("  Region Mesh: %d\n", ts.regions[j].mesh);
                }
                if (Args.print_sizes) {
                    if (Args.terse_output)
                        printf("%d:", ts.regions[j].size);
                    else
                        printf("  Region Size: %d\n", ts.regions[j].size);
                }
                if (Args.print_members) {
                    if (! Args.terse_output) {
                        printf("  Region Members:");
                        for (k = 0; k < ts.regions[j].size; k++)
                            printf(" %d", ts.regions[j].members[k]);
                        printf("\n");
                    }
                }
                if (Args.terse_output)
                    printf("\n");
            }
        }
        
    }
    
    return 0;
}

static const char *opt_string = "+ahl:Mm:n:st:";

static const struct option long_opts[] = {
    {"print_all", no_argument, NULL, 'a'},
    {"help", no_argument, NULL, 'h'},
    {"label", required_argument, NULL, 'l'},
    {"print_labels", no_argument, &Args.print_labels, 1},
    {"print_members", no_argument, &Args.print_members, 1},
    {"mesh", required_argument, NULL, 'm'},
    {"print_meshes", no_argument, &Args.print_meshes, 1},
    {"number", required_argument, NULL, 'n'},
    {"print_numbers", no_argument, &Args.print_numbers, 1},
    {"print_sizes", no_argument, &Args.print_sizes, 1},
    {"terse", no_argument, &Args.terse_output, 1},
    {"timestep", required_argument, NULL, 't'},
    {"print_timesteps", no_argument, &Args.print_timesteps, 1},
    {NULL, no_argument, NULL, 0} 
};

void _process_opts(int argc, char **argv) {
    
    int opt = 0;
    int long_index = 0;
    
    // Initialize
    Args.num_timesteps = 0;
    Args.print_timesteps = 0;
    Args.num_labels = 0;
    Args.print_labels = 0;
    Args.num_meshes = 0;
    Args.print_meshes = 0;
    Args.num_numbers = 0;
    Args.print_numbers = 0;
    Args.print_members = 0;
    Args.print_sizes = 0;
    Args.terse_output = 0;
    
    while( (opt = getopt_long( argc, argv, opt_string, long_opts, &long_index )) != -1 ) {
        switch(opt) {
            case 'a':
                Args.print_timesteps = 1;
                Args.print_labels = 1;
                Args.print_meshes = 1;
                Args.print_numbers = 1;
                Args.print_sizes = 1;
                break;
            case 'l':
                _parse_range(optarg, &Args.num_labels, &Args.label_list);
                break;
            case 'm':
                _parse_range(optarg, &Args.num_meshes, &Args.mesh_list);
                break;
            case 'n':
                _parse_range(optarg, &Args.num_numbers, &Args.number_list);
                break;
            case 't':
                _parse_range(optarg, &Args.num_timesteps, &Args.timestep_list);
                break;
            case 0:
                break;
            case 'h':
                _display_usage(argv[0]);
                break;
            default:
                break;
        }
    }
    
    argc -= optind;
    if (argc < 1) {
        fprintf(stderr, "Must specify a filename or '-' to read stdin\n");
        _display_usage(argv[0]);
    }
    
    argv += optind;
    if (! strcmp(argv[0], "-"))
        Args.filename = strdup("STDIN");
    else
        Args.filename = strdup(argv[0]);
    
}

void _display_usage(char *prog) {
    printf("usage: %s [options] filename\n\n", prog);
    printf("Prints on stdout user-specified information from a regions files. The\n");
    printf("<filename> argument may be either a filename or '-' to indicate stdin\n");
    printf("contains the region data. This allows piping from connect_comp_regions\n");
    printf("which prints the region data on stdout\n\n");
    printf("Region Selection Options:\n\n");
    printf("All region selection options take a RANGE argument which is a comma- and\n");
    printf("dash-delimited list of integers. E.g. the RANGE 1,4,6-8,10-12 would include\n");
    printf("the following: 1,4,6,7,8,10,11,12\n\n");
    printf("  --label|-l RANGE    : Only consider regions with label numbers in RANGE\n");
    printf("  --mesh|-m RANGE     : Only consider regions with mesh numbers in RANGE\n");
    printf("  --number|-n RANGE   : Only consider regions with region numbers in RANGE\n");
    printf("  --timestep|-t RANGE : Only consider regions in the timesteps in RANGE\n\n");
    printf("Printing Options:\n\n");
    printf("  --print_all|-a    : The equivalent of \"--print_labels --print_meshes\n");
    printf("                      --print_numbers --print_sizes --print_timesteps\"\n");
    printf("  --print_labels    : Turns on printing of region labels\n");
    printf("  --print_members   : Turns on printing of region members\n");
    printf("  --print_meshes    : Turns on printing of region mesh numbers\n");
    printf("  --print_numbers   : Turns on printing of region numbers\n");
    printf("  --print_sizes     : Turns on printing of region sizes\n");
    printf("  --print_timesteps : Turns on printing of timesteps\n");
    exit(-1);
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
