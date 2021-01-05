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
#include "av_utils.h"
#include "displacements.h"

void get_displacement_var(FC_Dataset ds, char *displ_var, FC_Variable*** displ) {
    #ifdef HAVE_AVATAR_FCLIB
    FC_ReturnCode rc;
    int i;
    int num_mesh, num_step;
    FC_Mesh *seq_meshes;
    FC_Variable ***Cdispl;
    
    // get number of meshes and setup var arrays
    rc = fc_getMeshes(ds, &num_mesh, &seq_meshes);
    av_exitIfError(rc);
    Cdispl = (FC_Variable***)malloc(num_mesh*sizeof(FC_Variable**));
    
    // loop over meshes and get displacement components
    int displ_status = 0;
    for (i = 0; i < num_mesh; i++) {
        //fc_getSeqVariableByName(seq_meshes[i], var_name, &num_step, &seq_vars[i]);
        Cdispl[i] = (FC_Variable**)malloc( 3 * sizeof(FC_Variable*) );
        if (displ_var != NULL) {
            char xcomp[ strlen(displ_var) + 2 ];
            strcpy(xcomp, displ_var);
            strcat(xcomp, "X");
            rc = fc_getSeqVariableByName(seq_meshes[i], xcomp, &num_step, &Cdispl[i][0]);
            if (rc != AV_SUCCESS) {
                strcpy(xcomp, displ_var);
                strcat(xcomp, "x");
                rc = fc_getSeqVariableByName(seq_meshes[i], xcomp, &num_step, &Cdispl[i][0]);
                if (rc != AV_SUCCESS) {
                    fprintf(stderr, "Did not find a displacement variable named '%sX' or '%sx'\n",
                                    displ_var, displ_var);
                    exit(65);
                } else {
                    char ycomp[ strlen(displ_var) + 2 ];
                    strcpy(ycomp, displ_var);
                    strcat(ycomp, "y");
                    displ_status += fc_getSeqVariableByName(seq_meshes[i], ycomp, &num_step, &Cdispl[i][1]);
                    char zcomp[ strlen(displ_var) + 2 ];
                    strcpy(zcomp, displ_var);
                    strcat(zcomp, "z");
                    displ_status += fc_getSeqVariableByName(seq_meshes[i], zcomp, &num_step, &Cdispl[i][2]);                    
                }
            } else {
                char ycomp[ strlen(displ_var) + 2 ];
                strcpy(ycomp, displ_var);
                strcat(ycomp, "Y");
                displ_status += fc_getSeqVariableByName(seq_meshes[i], ycomp, &num_step, &Cdispl[i][1]);
                char zcomp[ strlen(displ_var) + 2 ];
                strcpy(zcomp, displ_var);
                strcat(zcomp, "Z");
                displ_status += fc_getSeqVariableByName(seq_meshes[i], zcomp, &num_step, &Cdispl[i][2]);
            }
            
            if ( displ_status != AV_SUCCESS ) {
                fprintf(stderr, "Did not find one or more of the displacement components for '%s'\n", displ_var);
                exit(65);
            }
        } else {
            if ( fc_getSeqVariableByName(seq_meshes[i], "DISPLX", &num_step, &Cdispl[i][0]) == AV_SUCCESS ) {
                displ_status += fc_getSeqVariableByName(seq_meshes[i], "DISPLY", &num_step, &Cdispl[i][1]);
                displ_status += fc_getSeqVariableByName(seq_meshes[i], "DISPLZ", &num_step, &Cdispl[i][2]);
                displ_var = av_strdup("DISPL");
            } else if ( fc_getSeqVariableByName(seq_meshes[i], "displ_x", &num_step, &Cdispl[i][0]) == AV_SUCCESS ) {
                displ_status += fc_getSeqVariableByName(seq_meshes[i], "displ_y", &num_step, &Cdispl[i][1]);
                displ_status += fc_getSeqVariableByName(seq_meshes[i], "displ_z", &num_step, &Cdispl[i][2]);
                displ_var = av_strdup("displ_");
            } else if ( fc_getSeqVariableByName(seq_meshes[i], "DISX", &num_step, &Cdispl[i][0]) == AV_SUCCESS ) {
                displ_status += fc_getSeqVariableByName(seq_meshes[i], "DISY", &num_step, &Cdispl[i][1]);
                displ_status += fc_getSeqVariableByName(seq_meshes[i], "DISZ", &num_step, &Cdispl[i][2]);
                displ_var = av_strdup("DIS");
            } else {
                fprintf(stderr, "Did not find the displacement variable\n");
                exit(65);
            }
            if ( displ_status != AV_SUCCESS ) {
                fprintf(stderr, "Did not find one or more of the displacement components\n");
                exit(65);
            }
        }
    }
    
/*    // Remove mesh/var and mesh/seq_var entries that don't exist
    num_seq_var = 0;
    for (i = 0; i < num_mesh; i++) {
        if (seq_vars[i] == NULL) {
            for (j = i + 1; j < num_mesh; j++) {
                seq_meshes[j-1] = seq_meshes[j];
                seq_vars[j-1] = seq_vars[j];
                Cdispl[j-1][0] = Cdispl[j][0];
                Cdispl[j-1][1] = Cdispl[j][1];
                Cdispl[j-1][2] = Cdispl[j][2];
            }
        }
        else 
            num_seq_var++;
    }
*/    
    // Combine component displacements
    *displ = (FC_Variable**)malloc( num_mesh * sizeof(FC_Variable*) );
    for (i = 0; i < num_mesh; i++) {
        rc = fc_mergeComponentSeqVariables(3, num_step, Cdispl[i], displ_var, FC_MT_VECTOR, &(*displ)[i]);
        av_exitIfError(rc);
    }
    for (i = 0; i < num_mesh; i++) {
        free(Cdispl[i]);
    }
    free(Cdispl);
    #else
    av_printfErrorMessage("This function (get_displacement_var) requires fclib 1.6.1.\n"
        "Put fclib source to be in avatar/util/ and then rebuild.");
    #endif
}
