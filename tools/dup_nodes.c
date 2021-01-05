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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>

#include "netcdf.h"
#include "array.h"

int _process_options(int argc, char **argv);
int _get_nodes_for_proc(int proc, size_t *num_int_node, int **int_nodes, size_t *num_bor_node, int **bor_nodes);
int _get_all_nodes_for_proc(size_t *num_int_node, int **int_nodes, size_t *num_bor_node, int **bor_nodes);
void usage(char *progname);

char *filestem = NULL;
char *extension = NULL;
char *out_filestem = NULL;
int num_procs = 0;
int proc = -1;
int write_to_files = 1;
int ncid;
int force_node_mapi = -1;
int force_node_mapb = -1;

int main( int argc, char *argv[] ) {
    
    int i, j;
    size_t num_int_node;
    size_t num_bor_node;
    int *int_nodes, *bor_nodes;
    int rc;
    // Read input options
    if (_process_options(argc, argv)) {
        usage(argv[0]);
        return -1;
    }
    
    // Figure out where we should start and end.
    int start_proc = proc == -1 ? 0 : proc;
    int end_proc = proc == -1 ? num_procs-1 : proc;
    for (i = start_proc; i <= end_proc; i++) {
        int size = strlen(filestem) + strlen(extension) + 3 + 2*((int)log10(num_procs) + 1) + 1;
        char datafile[size];
        sprintf(datafile, "%s.%s.%d.%d", filestem, extension, num_procs, i);
        //printf("Processing %s\n", datafile);
                
        // Open as netcdf file
        rc = nc_open(datafile, NC_SHARE, &ncid);
        if (rc) {
            fprintf(stderr, "Could not open %s\n", datafile);
            return -1;
        }
        rc = _get_nodes_for_proc(i, &num_int_node, &int_nodes, &num_bor_node, &bor_nodes);
        //printf("Found %3d/%3d internal/border nodes\n", num_int_node, num_bor_node);
        rc = nc_close(ncid);
        
        if (! write_to_files) {
            printf("\nPARTITION FILE %d\nInternal Nodes\n", i);
            for (j = 0; j < num_int_node; j++)
                printf("%d ", int_nodes[j]);
            printf("\nBorder Nodes\n");
            for (j = 0; j < num_bor_node; j++)
                printf("%d ", bor_nodes[j]);
            printf("\n");
        } else {
            if (out_filestem != NULL)
                size = strlen(out_filestem) + strlen(extension) + 3 + 2*((int)log10(num_procs) + 1) + 1;
            char int_nodefile[size+5];
            char bor_nodefile[size+5];
            if (out_filestem != NULL) {
                sprintf(int_nodefile, "%s.%s.%d.%d.int", out_filestem, extension, num_procs, i);
                sprintf(bor_nodefile, "%s.%s.%d.%d.bor", out_filestem, extension, num_procs, i);
            } else {
                sprintf(int_nodefile, "%s.%s.%d.%d.int", filestem, extension, num_procs, i);
                sprintf(bor_nodefile, "%s.%s.%d.%d.bor", filestem, extension, num_procs, i);
            }
            
            FILE *int_file;
            FILE *bor_file;
            int_file = fopen(int_nodefile, "w");
            bor_file = fopen(bor_nodefile, "w");
            
            for (j = 0; j < num_int_node; j++)
                fprintf(int_file, "%d\n", int_nodes[j]);
            for (j = 0; j < num_bor_node; j++)
                fprintf(bor_file, "%d\n", bor_nodes[j]);

            fclose(int_file);
            fclose(bor_file);
        }
    }

    
    return 0;
}

int _get_nodes_for_proc(int prc, size_t *num_int_node, int **int_nodes, size_t *num_bor_node, int **bor_nodes) {
    
    static int *global_int_nodes, *global_bor_nodes;
    static int num_global_int_nodes = 0;
    static int num_global_bor_nodes = 0;
    static size_t num_nodes_global;
    static int *node_counts;
    static int parsed_all = 0;
    
    int i;
    unsigned int j;
    
    int error;
    int dimid, varid;
    size_t num_nodes;
    int *node_num_map, *node_mapi, *node_mapb;
    
    int use_bor = 1;
    int use_int = 1;
    
    error = nc_inq_dimid(ncid, "num_nodes_global", &dimid) || nc_inq_dimlen(ncid, dimid, &num_nodes_global);
    if (error) {
        fprintf(stderr, "num_nodes_global: %s\n", nc_strerror(error));
        return -1;
    }
    
    error = nc_inq_dimid(ncid, "num_nodes", &dimid) || nc_inq_dimlen(ncid, dimid, &num_nodes);
    if (error) {
        fprintf(stderr, "num_nodes: %s\n", nc_strerror(error));
        return -1;
    }
    error = nc_inq_dimid(ncid, "num_int_node", &dimid) || nc_inq_dimlen(ncid, dimid, num_int_node);
    if (error) {
        //fprintf(stderr, "num_int_node: %s\n", nc_strerror(error));
        use_int = 0;
    }
    // If this = num_nodes, don't rely on it
    if (num_int_node == num_nodes)
        use_int = 0;
        
    error = nc_inq_dimid(ncid, "num_bor_node", &dimid) || nc_inq_dimlen(ncid, dimid, num_bor_node);
    if (error) {
        //fprintf(stderr, "num_bor_node: %s\n", nc_strerror(error));
        use_bor = 0;
    }
    node_num_map = (int *)calloc(num_nodes, sizeof(int));
    error = nc_inq_varid(ncid, "node_num_map", &varid) || nc_get_var_int(ncid, varid, node_num_map);
    if (error) {
        fprintf(stderr, "node_num_map: %s\n", nc_strerror(error));
        return -1;
    }
    
    // Check for forced values
    // -1 => no forcing -- use values as determined by what info the file contains
    //  0 => force off
    //  1 => force on
    use_int = force_node_mapi > -1 ? force_node_mapi : use_int;
    use_bor = force_node_mapb > -1 ? force_node_mapb : use_bor;

    if (use_int && use_bor) {
        
        // This is the easy case and is applicable to mesh files output by the nemesis tools
        // Both mappings are in the netcdf so we just need to read them.
        
        node_mapi = (int *)calloc(*num_int_node, sizeof(int));
        *int_nodes = (int *)calloc(*num_int_node, sizeof(int));
        node_mapb = (int *)calloc(*num_bor_node, sizeof(int));
        *bor_nodes = (int *)calloc(*num_bor_node, sizeof(int));
        
        error = nc_inq_varid(ncid, "node_mapi", &varid) || nc_get_var_int(ncid, varid, node_mapi);
        if (error) {
            fprintf(stderr, "node_mapi: %s\n", nc_strerror(error));
            return -1;
        }
        error = nc_inq_varid(ncid, "node_mapb", &varid) || nc_get_var_int(ncid, varid, node_mapb);
        if (error) {
            fprintf(stderr, "node_mapb: %s\n", nc_strerror(error));
            return -1;
        }
        
        for (i = 0; i < *num_int_node; i++) {
            (*int_nodes)[i] = node_num_map[node_mapi[i] - 1];
        }
        for (i = 0; i < *num_bor_node; i++)
            (*bor_nodes)[i] = node_num_map[node_mapb[i] - 1];
        
    } else {
    // USE THIS AS FALLBACK IF BOTH INT AND BOR ARE NOT GIVEN
    //} else if (! use_int && ! use_bor) {
        // Warn the user
        fprintf(stderr, "WARNING: Not enough information in each file. Must parse ALL files.\n");
        
        // Re-initialize since they may contain erroneous values
        *num_int_node = 0;
        *num_bor_node = 0;
        
        // In this case our only option is to read all the parition files and look for duplicated
        // nodes which are border nodes.
        
        if (! parsed_all) {
            // Only need to do this once since we'll reuse the global lists for each processor.
            parsed_all = 1;
            
            // global_nodes is a list of all nodes. It includes repeated border nodes.
            int *global_nodes;
            global_nodes = (int *)calloc(num_nodes, sizeof(int));
            // This is the offset into global_nodes for each partition file.
            int offset = 0;
            
            // We've already read this partition file so just use it.
            for (i = 0; i < num_nodes; i++)
                global_nodes[i+offset] = node_num_map[i];
            offset += num_nodes;
            
            // Read all partition files
            for (i = 0; i < num_procs; i++) {
                // Skip the one we've already handled
                if (i == prc)
                    continue;

                int size = strlen(filestem) + strlen(extension) + 3 + 2*((int)log10(num_procs) + 1) + 1;
                char datafile[size];
                sprintf(datafile, "%s.%s.%d.%d", filestem, extension, num_procs, i);
                if (nc_open(datafile, NC_SHARE, &ncid)) {
                    fprintf(stderr, "Could not open %s\n", datafile);
                    return -1;
                }
                
                // Read this paritition's num_nodes and node_num_map
                int did;
                size_t nn;
                int *nnm;
                error = nc_inq_dimid(ncid, "num_nodes", &did) || nc_inq_dimlen(ncid, did, &nn);
                if (error) {
                    fprintf(stderr, "num_nodes: %s\n", nc_strerror(error));
                    return -1;
                }
                nnm = (int *)calloc(nn, sizeof(int));
                error = nc_inq_varid(ncid, "node_num_map", &varid) || nc_get_var_int(ncid, varid, nnm);
                if (error) {
                    fprintf(stderr, "node_num_map: %s\n", nc_strerror(error));
                    return -1;
                }
                
                // Reallocate memory for global_nodes and add current partition's node_num_map values
                global_nodes = (int *)realloc(global_nodes, (offset + nn) * sizeof(int));
                for (j = 0; j < nn; j++)
                    global_nodes[j+offset] = nnm[j];
                offset += nn;
                
                free(nnm);
            }
            
            // Go through global_nodes and count the occurances of each node number.
            // If a node occurs a second time, it's a border node so update border node count
            node_counts = (int *)calloc(num_nodes_global, sizeof(int));
            for (i = 0; i < offset; i++) {
                node_counts[global_nodes[i] - 1]++;
                if (node_counts[global_nodes[i] - 1] == 2) {
                    num_global_bor_nodes++;
                    //printf("Found dup node %4d: %4d\n", num_global_bor_nodes, global_nodes[i]);
                }
            }
            // Now that we know how many nodes are in each list, allocate and populate the lists.
            num_global_int_nodes = num_nodes_global - num_global_bor_nodes;
            global_int_nodes = (int *)calloc(num_global_int_nodes, sizeof(int));
            global_bor_nodes = (int *)calloc(num_global_bor_nodes, sizeof(int));
            int in = 0;
            int bn = 0;
            for (i = 0; i < num_nodes_global; i++) {
                if (node_counts[i] > 1) {
                    //printf("Node %d is a border node\n", i+1);
                    global_bor_nodes[bn++] = i + 1;
                } else if (node_counts[i] == 1) {
                    //printf("Node %d is an internal node\n", i+1);
                    global_int_nodes[in++] = i + 1;
                } else
                    fprintf(stderr, "Invalid node number\n");
            }
        }
        
        // For now, just make (int|bor)_nodes the same size as global lists. May want to think about memory usage later
        *int_nodes = (int *)calloc(num_global_int_nodes, sizeof(int));
        *bor_nodes = (int *)calloc(num_global_bor_nodes, sizeof(int));

        // Step through the current partition's node list and check node_counts
        for (i = 0; i < num_nodes; i++) {
            if (node_counts[node_num_map[i]-1] == 1) {
                (*int_nodes)[*num_int_node] = node_num_map[i];
                (*num_int_node)++;
            } else {
                (*bor_nodes)[*num_bor_node] = node_num_map[i];
                (*num_bor_node)++;
            }
                
        }
        
    }
    /* DON'T EVEN TRY THESE LAST TWO OPTIONS SINCE THEY ARE UNRELIABLE
    } else if (use_int) {
        // Warn the user
        fprintf(stderr, "WARNING: Using the node_mapi netcdf var which is not guaranteed to be correct\n");
        
        node_mapi = (int *)calloc(*num_int_node, sizeof(int));
        *int_nodes = (int *)calloc(*num_int_node, sizeof(int));
        
        // Read node_mapi
        error = nc_inq_varid(ncid, "node_mapi", &varid) || nc_get_var_int(ncid, varid, node_mapi);
        if (error) {
            fprintf(stderr, "node_mapi: %s\n", nc_strerror(error));
            return -1;
        }
        
        // In this case, num_int_node is set to num_nodes in the netcdf file.
        // So, assume that the valid node nubmers are in monotonically increasing order
        // and read node numbers until we violate this assumption. This is the actual
        // num_int_node.
        for (i = 0; i < *num_int_node; i++) {
            if (i > 0 && node_mapi[i] < node_mapi[i-1]) {
                if (node_mapi[i] == 0) {
                    // This is most likely valid
                } else if (node_mapi[i] == 1) {
                    fprintf(stderr, "WARNING: The one unlikely but ambiguous case has been encountered.\n");
                    fprintf(stderr, "Node 1 has been found but this may or may not be a valid internal node.\n");
                }
                *num_int_node = i;
                break;
            }
            if (node_mapi[i] > 0)
                (*int_nodes)[i] = node_num_map[node_mapi[i]-1];
        }

        // Any node in node_num_map but not in int_nodes is a border node.
        *num_bor_node = num_nodes - *num_int_node;
        array_diff(node_num_map, num_nodes, *int_nodes, *num_int_node, bor_nodes);
        
    } else if (use_bor) {
        // Warn the user
        fprintf(stderr, "WARNING: Using the node_mapb netcdf var which is not guaranteed to be correct\n");
        
        // Have not run across this case but assume it's possible and treat it
        // analogous to the case of (use_int && ! use_bor)
        
        node_mapb = (int *)calloc(*num_bor_node, sizeof(int));
        *bor_nodes = (int *)calloc(*num_bor_node, sizeof(int));
        
        error = nc_inq_varid(ncid, "node_mapb", &varid) || nc_get_var_int(ncid, varid, node_mapb);
        if (error) {
            fprintf(stderr, "node_mapb: %s\n", nc_strerror(error));
            return -1;
        }
        
        for (i = 0; i < *num_bor_node; i++) {
            if (i > 0 && node_mapb[i] < node_mapb[i-1]) {
                if (node_mapb[i] == 0) {
                    // This is most likely valid
                } else if (node_mapb[i] == 1) {
                    fprintf(stderr, "WARNING: The one unlikely but ambiguous case has been encountered.\n");
                    fprintf(stderr, "Node 1 has been found but this may or may not be a valid border node.\n");
                }
                *num_bor_node = i;
                break;
            }
            (*bor_nodes)[i] = node_num_map[node_mapb[i]-1];
        }
        
        // Any node in node_num_map but not in int_nodes is a border node.
        *num_int_node = num_nodes - *num_bor_node;
        array_diff(node_num_map, num_nodes, *bor_nodes, *num_bor_node, int_nodes);

    }*/
        
    return 0;
}

static struct option longopts[] = {
    {"extension", required_argument, NULL, 'e'},
    {"filestem", required_argument, NULL, 'f'},
    {"force_node_mapi_on", no_argument, &force_node_mapi, 1},
    {"force_node_mapi_off", no_argument, &force_node_mapi, 0},
    {"force_node_mapb_on", no_argument, &force_node_mapb, 1},
    {"force_node_mapb_off", no_argument, &force_node_mapb, 0},
    {"num_procs", required_argument, NULL, 'n'},
    {"output_filename", required_argument, NULL, 'o'},
    {"proc", required_argument, NULL, 'p'},
    {"write_to_term", no_argument, &write_to_files, 0},
    {NULL,0,NULL,0}
};

int _process_options(int argc, char **argv) {
    int o;
    //char *optarg;
    while ( (o = getopt_long(argc, argv, "e:f:n:o:p", longopts, NULL )) != -1 ) {
        switch (o) {
            case 'e':
                extension = strdup(optarg);
                break;
            case 'f':
                filestem = strdup(optarg);
                break;
            case 'n':
                num_procs = atoi(optarg);
                break;
            case 'o':
                out_filestem = strdup(optarg);
                break;
            case 'p':
                proc = atoi(optarg);
                break;
            case 0:
                break;
            default:
                usage(argv[0]);
        }
    }
    
    if (filestem == NULL || extension == NULL || num_procs == 0)
        return -1;
    if (proc > -1 && proc >= num_procs)
        return -1;
    
    return 0;
}

void usage(char *progname) {
    printf("Usage: %s -f filestem -e extension -n total_procs [ -p proc_num ] [ -w ]\n\n", progname);
    printf("  -f filestem    : These three arguments define the filename(s) that are used\n");
    printf("  -e extension   : to discover the border and internal nodes. The filename(s)\n");
    printf("  -n total_procs : are 'filestem.extension.total_procs.N' where N is the par-\n");
    printf("                   ticular processor file of interest (as defined by the -p\n");
    printf("                   option\n");
    printf("  -o out_filestem: Specify an alternate filestem for the output files.\n");
    printf("                   The default is the same as the input filestem.\n");
    printf("  -p proc_num    : If specified, only the border and internal nodes for this\n");
    printf("                   processor file are discovered. If not specified, the list of\n");
    printf("                   global border and internal nodes are discovered\n");
    printf("                   NOTE: This number is zero-based so it ranges from\n");
    printf("                         0 -> total_procs-1\n");
    printf("  --write_to_term: Default behavious is to write the nodes lists to files. This\n");
    printf("                   option causes the lists to be printed on STDOUT\n");
}