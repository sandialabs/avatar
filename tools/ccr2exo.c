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

#include "exodusII.h"
#include "fc.h"
#include "array.h"
#include "regions.h"

int main(int argc, char** argv) {
    FC_ReturnCode rc;
    int i, j, k;
    int num_mesh, num_seq_var, num_step;
    FC_Dataset ds;
    FC_Sequence *sequence;
    FC_Mesh *seq_meshes;
    FC_Variable** seq_vars; // array of seq vars (num_mesh long)
    char* regions_filename;
    char* output_filename;
    char* exodus_filename = NULL;
    FC_VerbosityLevel verbose_level = FC_QUIET; 
    FC_AssociationType assoc;
    int num_comp;
    int *num_datapoints;
    double ***region_number_data;
    FC_Variable** region_number_vars;
    FC_FileIOType file_type;
    FC_Dataset ds_new;

    FILE *regions_file;
    timestep *timesteps;
    region_file_metadata metadata;


    // handle arguments
    if (argc < 3) {
        usage:
        printf("usage: %s [options] regions_test_file output_exodus\n", argv[0]);
        printf("options: \n");
        printf("   -dataset filename: use filename as the original exodus file instead\n");
        printf("                      of the one named in the regions file\n");
        printf("   -h               : print this help message\n");
        printf("   -v               : verbose: print warning and error messages\n");
        printf("   -V               : very verbose: print log and error messages\n");
        printf("\n");
        printf("Writes a new exodus dataset which includes region number as a variable.\n");
        printf("\n");
        printf("Example: %s data.ex2 data.regions data.regions.ex2\n", argv[0]);
        printf("\n");
        exit(-1);
    }
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-dataset")) {
            exodus_filename = argv[i+1];
            i++;
        } else if (!strcmp(argv[i], "-v")) {
            verbose_level = FC_WARNING_MESSAGES;
        } else if (!strcmp(argv[i], "-V")) {
            verbose_level = FC_LOG_MESSAGES;
        } else if (!strncmp(argv[i], "-h", 2) || !strncmp(argv[i], "-", 1))
            goto usage;
        else {
            if (i+1 >= argc)
                goto usage;
            regions_filename = argv[i];
            output_filename = argv[i+1];
            i+=1;
        }
    }
    if (!regions_filename || !output_filename)
        goto usage;
    
    // Read regions file
    regions_file = fopen(regions_filename, "r");
    if (regions_file == NULL) {
        fprintf(stderr, "Error: Unable to read regions file '%s'\n", regions_filename);
        exit(8);
    }
    rc = read_region_data(regions_file, &timesteps, &metadata);
    fc_exitIfError(rc);
//    printf("Region file was created for dataset = %s using variable = %s and threshold = %s %f\n",
//           metadata.file_name, metadata.var_name, metadata.threshold_op, metadata.threshold_val);
  
    // init library and load dataset
    rc = fc_setLibraryVerbosity(verbose_level);
    fc_exitIfError(rc);
    rc = fc_initLibrary();
    fc_exitIfError(rc);
    if (exodus_filename == NULL) {
        rc = fc_loadDataset(metadata.file_name, &ds);
        fc_exitIfError(rc);
    } else {
        rc = fc_loadDataset(exodus_filename, &ds);
        fc_exitIfError(rc);
    }

    // get number of meshes and setup var arrays
    rc = fc_getMeshes(ds, &num_mesh, &seq_meshes);
    fc_exitIfError(rc);
    seq_vars = (FC_Variable**)malloc(num_mesh*sizeof(FC_Variable*));

    // loop over meshes
    for (i = 0; i < num_mesh; i++) {
        FC_Variable **sv;
        int nsv, *nspv;
        fc_getSeqVariableByName(seq_meshes[i], metadata.var_name, &nsv, &nspv, &sv);
        num_step = nspv[0];
        seq_vars[i] = sv[0];
    }

    // Remove mesh/var and mesh/seq_var entries that don't exist
    num_seq_var = 0;
    for (i = 0; i < num_mesh; i++) {
        if (seq_vars[i] == NULL) {
            for (j = i + 1; j < num_mesh; j++) {
                seq_meshes[j-1] = seq_meshes[j];
                seq_vars[j-1] = seq_vars[j];
            }
        }
        else 
            num_seq_var++;
    }
  
    // did we find anything?
    if (num_seq_var < 1)
        fc_exitIfErrorPrintf(FC_ERROR, "Failed to find variable '%s' in dataset "
                                       "'%s'", metadata.var_name, metadata.file_name);

    if (num_seq_var > 0) {
        FC_MathType mathtype;
        FC_DataType datatype;

        num_datapoints = (int *)malloc(num_seq_var * sizeof(int));
        sequence = (FC_Sequence *)malloc(num_seq_var * sizeof(FC_Sequence));

        region_number_data = (double ***)malloc(num_seq_var * sizeof(double *));
        if (region_number_data == NULL) {
            fc_exitIfError(FC_MEMORY_ERROR);
        }
        region_number_vars = (FC_Variable **)malloc(num_seq_var * sizeof(FC_Variable *));
        if (region_number_vars == NULL) {
            fc_exitIfError(FC_MEMORY_ERROR);
        }

        for (i = 0; i < num_seq_var; i++) {
            fc_getSequenceFromSeqVariable(num_step, seq_vars[i], &sequence[i]);
            fc_exitIfError(rc);

            // Get metadata for variable
            fc_getVariableInfo(seq_vars[i][0], &num_datapoints[i], &num_comp, &assoc, &mathtype, &datatype);
            
            // Allocate space for the new variable data
            region_number_data[i] = (double **)malloc(num_step * sizeof(double *));
            if (region_number_data[i] == NULL) {
                fc_exitIfError(FC_MEMORY_ERROR);
            }
            for (j = 0; j < num_step; j++) {
                region_number_data[i][j] = malloc(num_datapoints[i] * sizeof(double));
                // Initialize variable data
                for (k = 0; k < num_datapoints[i]; k++) {
                    region_number_data[i][j][k] = -1;
                }
            }
        }

        // Set region number data from timesteps structures
        for (i = 0; i < metadata.num_timesteps; i++) {
            int step = timesteps[i].timestep;
            for (j = 0; j < timesteps[i].num_regions; j++) {
                int mesh = timesteps[i].regions[j].mesh;
                int label = timesteps[i].regions[j].label;
                for (k = 0; k < timesteps[i].regions[j].size; k++) {
                    // region members are 1-based but fclib is 0-based so subtract 1
                    int node = timesteps[i].regions[j].members[k] - 1;
                    //printf("Setting node %d at step %d on mesh %d to %d\n", node, step, mesh, label);
                    region_number_data[mesh][step][node] = label;
                }
            }
        }
                
        for (i = 0; i < num_seq_var; i++) {
            // Create new variable
            int temp_num_step;
            rc = fc_createSeqVariable(seq_meshes[i], sequence[i], "dd_region_number", &temp_num_step,
                                      &region_number_vars[i]);
            fc_exitIfError(rc);
            for (j = 0; j < num_step; j++) {
                rc = fc_setVariableDataPtr(region_number_vars[i][j], num_datapoints[i], 
                                           num_comp, assoc, mathtype, FC_DT_DOUBLE, region_number_data[i][j]);
                fc_exitIfError(rc);
            }
        }
        
    }

    free_region_data(timesteps, metadata);
    
    // copy the dataset & write copy to disk
    rc = fc_getDatasetFileIOType(ds, &file_type);
    fc_exitIfErrorPrintf(rc, "Failed to get file IO type");

    char dataset_name[strlen(output_filename)];
    strcpy(dataset_name, output_filename);
    // Adjust dataset name for exodus if name is longer than MAX_LINE_LENGTH
    if (file_type == FC_FT_EXODUS) {
        if (strlen(dataset_name) > MAX_LINE_LENGTH) {
            char temp_name[strlen(dataset_name)];
            strcpy(temp_name, dataset_name);
            strcpy(dataset_name, "...");
            int count = strlen(temp_name);
            char *chunk;
            char sep[1] = "/";
            chunk = strtok(temp_name, sep);
            while (chunk != NULL) {
                chunk = strtok(NULL, sep);
                if (chunk == NULL) {
                    break;
                }
                // the '+1' allows for the '/' which will be added back to the string
                count -= strlen(chunk) + 1;
                // the -3 allows for the '...' at the beginning of dataset_name
                if (count < MAX_LINE_LENGTH - 3) {
                    strcat(dataset_name, "/");
                    strcat(dataset_name, chunk);
                }
            }
        }
    }

    printf("Copying dataset to new name: %s\n", dataset_name);
    rc = fc_copyDataset(ds, dataset_name, &ds_new);
    fc_exitIfErrorPrintf(rc, "Failed to copy dataset");

    printf("Printing dataset to %s\n", output_filename);
    rc = fc_writeDataset(ds_new, output_filename, file_type);
    fc_exitIfErrorPrintf(rc, "Failed to write dataset");
    
    // cleanup
    fc_finalLibrary();

    free(seq_meshes);
    for (i = 0; i < num_seq_var; i++) {
        free(seq_vars[i]);
        free(region_number_data[i]);
        free(region_number_vars[i]);
    }
    free(seq_vars);
    free(region_number_data);
    free(region_number_vars);
    free(num_datapoints);
    free(sequence);
  
    exit(0);
}


