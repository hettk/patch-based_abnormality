//
//Optimized version with mthreaded Seg
//
//patchmatch.c //
//
//Main program.
//Uses the patchmatch algorithm to return the Nearest Neighbor Field giving
//the correspondance between two images.
//
//
//INPUTS: - a: Subject to study
//        - b: Stack
//        - mask: Pre determined region where to perform the algorithm
//        - mask_dil: Extended mask to correct side effects
//        - limit: Number of iterations not to exceed
//        - knn_nbr: Number of k nearest neighbor to return (1, 9, or 27).
//        - multiple_pm_nbr: Multiple Patchmatch
//        - color: To 1 if the images are RGB.
//        - multithreading_multiple_pm: To 1 if every multiple patchmatchtch is send to a dedicated thread.
//        - N_thread_A: 1 4 or 8.
//        - a_real_seg: Manual expert segmentaton of the subject to study
//        - library_real_seg: Manual expert segmentaton of the stack
//        - list_lab_mask: List of label correspondences
//        - label_to_ind: Give the subscript of a label
//
//
//OUTPUTS: - nnf: NNF map. Matches fo A voxels
//         - nnfd: NNFD map distance. dima dimensions. Gives the best
//                  distance computed according to the data in nnf
//         - rmsd: Root-to-Mean Square Deviation. Vector 1xlimit giving
//                  the deviation between nnf_i and nnf_i-1
//         - a_seg_th: Our segmented subject A
//
//
//Structure of the NNF & NNFD MAPS:
//
//3D nnf matrix: (ha, wa, da, 3, knn_nbr*multiple_pm_nbr)
/* nnf(:,:,:,1,:) => x (w) */
/* nnf(:,:,:,2,:) => y (h) */
//nnf(:,:,:,3,:) => z (d)
//nnf(:,:,:,:,1)
//    ...
//nnf(:,:,:,:,knn_nbr+1) => First set of matches (First Patchmatchtchmatch)
//nnf(:,:,:,:,knn_nbr+2)
//    ...
//nnf(:,:,:,:,2*(knn_nbr+1)) => Second set of matches (Second patchmatch)
//    ...
//
//
//3D nnfd matrix: (ha, wa, da, knn_nbr*multiple_pm_nbr)
//nnfd(:,:,:,1) => First set of distances (First Patchmatch)
//nffd(:,:,:,2) => Second set of dsitances (Second patchmatch)
//    ...
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"
#include <semaphore.h>
#include <unistd.h>

#ifndef MAX
#define MAX(a, b) ((a)>(b)?(a):(b))
#endif

#ifndef MIN
#define MIN(a, b) ((a)<(b)?(a):(b))
#endif

#define EPSILON 0.0000001

#ifndef enc_type
#define enc_type int
#endif


typedef struct{
    unsigned char   *b_real_seg;
    unsigned char   *mask;
    double          *a_seg_multilabel;
    enc_type        *nnf;
    float           *nnfd;
    int             knn_total;
    int             ha;
    int             wa;
    int             da;
    int             hb;
    int             wb;
    unsigned char   *label_to_ind;
    int             patch_w;
    int             ini;
    int             fin;
    sem_t           *mutex;
}myargument;







int opal_rand(unsigned long* next)
{
    (*next) = (*next) * 1103515245 + 12345;
    return (unsigned int)((*next)/65536) % 32768;
}

float ssd_3D(float * subject, float *library, int ha, int wa,  int hb, int wb,
        int ay, int ax, int az, int by, int bx, int bz, int pw) {
    
    float d_tot = 0, d = 0;
    for (int z = -pw; z <= pw; z++) {
        for (int x = -pw; x <= pw; x++) {
            for (int y = -pw; y <= pw; y++) {
                d = subject[(ay+y)+(ha*(ax+x))+(ha*wa*(az+z))];
                d -= library[(by+y)+(hb*(bx+x))+(hb*wb*(bz+z))] ;
                d_tot += d*d;
            }
        }
    }
    d_tot /= ((2*pw+1)*(2*pw+1)*(2*pw+1));
    
    return d_tot;
}




typedef struct{
    float* subject;
    float* library;
    unsigned char* mask;
    unsigned char* mask_path;
    int nb_vox;
    enc_type* nnf;
    float *nnfd;
    int wa;
    int ha;
    int da;
    int wb;
    int hb;
    int db;
    int iter;
    int patch_radius;
    int multiple_pm_nbr;
    int proc_nbr;
    int mag_init;
    int offset_mp;
    unsigned long next;
    sem_t* mutex_list;
    int * nnf_list_pos;
    int use_mutex;
}opal_struct;

void* opal(void *arg)
{
    opal_struct inputs = *(opal_struct*) arg;
    float *subject     = inputs.subject;
    float *library     = inputs.library;
    unsigned char *mask        = inputs.mask;
    unsigned char *mask_path   = inputs.mask_path;
    int nb_vox         = inputs.nb_vox;
    enc_type  *nnf     = inputs.nnf;
    float *nnfd        = inputs.nnfd;
    int wa             = inputs.wa;
    int ha             = inputs.ha;
    int da             = inputs.da;
    int wb             = inputs.wb;
    int hb             = inputs.hb;
    int db             = inputs.db;
    int patch_radius   = inputs.patch_radius;
    int mag_init       = inputs.mag_init;
    int iter          = inputs.iter;
    int offset_mp      = inputs.offset_mp;
    unsigned long next = inputs.next;
    int pw = patch_radius;
    sem_t* mutex_list = inputs.mutex_list;
    int * nnf_list_pos = inputs.nnf_list_pos;
    int multiple_pm_nbr = inputs.multiple_pm_nbr;
    int proc_nbr = inputs.proc_nbr;
    int use_mutex = inputs.use_mutex;
    
    // Initialize with random nearest neighbor field (NNF).
    // Effective width and height (possible upper left corners of patches).
    //int aew = wa-patch_radius, aeh = ha-patch_radius, aed = da-patch_radius;
    int ha_wa = ha*wa;
    int hb_wb = hb*wb;
    int size_a = ha_wa*da;
    int size_ax2 = 2*size_a;
    int size_ax3 = 3*size_a;
    
    int nbtemplates = db/da;
    int rand_val = 0;
    float dist = 0;
    float d_tot; //d 
    //float band_add, band_cut, band_add_d, band_cut_d;
    
    //RANDOM INITIALIZATION
    if (iter == 0) {
        for (int nn=0; nn<nb_vox; nn++) {
            
            int ax = mask_path[nn];
            int ay = mask_path[nn + nb_vox];
            int az = mask_path[nn + nb_vox*2];
            
            int pos_current_a = ha*ax + ha_wa*az + ay;
            
            //Box limits
            int pos_current = pos_current_a  + offset_mp*4;
            int pos_current_nnfd = pos_current_a + offset_mp;
            
            int xmin = MAX(ax-mag_init, pw);
            int xmax = MIN(ax+mag_init+1,wa-pw);
            int ymin = MAX(ay-mag_init, pw);
            int ymax = MIN(ay+mag_init+1,ha-pw);
            int zmin = MAX(az-mag_init, pw);
            int zmax = MIN(az+mag_init+1,da-pw);
            
            int init_ = 1;
            while (init_) {
                /* Random match within the box around the search area */
                /* of size patch_radius x patch_radius x patch_radius */
                rand_val = opal_rand(&next);
                int bx = xmin+rand_val%(xmax-xmin);
                rand_val = opal_rand(&next);
                int by = ymin+rand_val%(ymax-ymin);
                rand_val = opal_rand(&next);
                int bz = zmin+rand_val%(zmax-zmin);
                rand_val = opal_rand(&next);
                int tp = rand_val%nbtemplates;
                int zp = bz+(tp*da);
                
                int pos_b = by + bz*hb + zp*hb*wb;
                
                //mutex table
                int ok = 0, ok2 = 0;
                for (ok = 0; ok<multiple_pm_nbr; ok++) {
                    if (use_mutex)
                        sem_wait(&(mutex_list[pos_current_a + ok*size_a]));
                    if (pos_b == nnf_list_pos[pos_current_a + ok*size_a])
                        break;
                }
                if (ok == multiple_pm_nbr) { // not used in the list
                    //Distance computation
                    dist = ssd_3D(subject, library, ha, wa, hb, wb, ay, ax, az, by, bx, zp, pw);
                    
                    nnf[pos_current] = bx;
                    nnf[pos_current + size_a] = by;
                    nnf[pos_current + size_ax2] = bz;
                    nnf[pos_current + size_ax3] = tp;
                    nnfd[pos_current_nnfd] = dist;
                    nnf_list_pos[pos_current_a + proc_nbr*size_a] = pos_b;
                    
                    init_ = 0;
                }
                if (use_mutex) {
                    for (ok2 = 0; ok2<=MIN(ok,multiple_pm_nbr-1); ok2++) //free the mutex
                        sem_post(&(mutex_list[pos_current_a + ok2*size_a]));
                }
                
            }
            
            
        }
    }
    
    int count = 0;
    while ((count < iter)) {
        
        
        int nn_change = 1;
        int nn_start = 0;
        int nn_end = nb_vox-1;
        //int off = 1;
        
        if (count % 2 == 1) {
            nn_start = nb_vox-1;
            nn_end = 0;
            nn_change = -1;
            //off = 0;
        }
        
        
        for (int nn=nn_start; nn!=nn_end; nn += nn_change) {
            
            int ax = mask_path[nn];
            int ay = mask_path[nn + nb_vox];
            int az = mask_path[nn + nb_vox*2];
            
            int pos_current_a = ha*ax + ha_wa*az + ay;
            
            
            /* label_mask = mask[pos_current_a]; */
            //Current position in nnf
            int pos_current = pos_current_a + 4*offset_mp;
            int pos_current_nnfd =  pos_current_a + offset_mp;
            
            // Current (best) guess.
            int xbest = nnf[pos_current];
            int ybest = nnf[pos_current + size_a];
            int zbest = nnf[pos_current + size_ax2];
            int tbest = nnf[pos_current + size_ax3];
            float dbest = nnfd[pos_current_nnfd];
            float *bestd = &dbest;
            int *bestx = &xbest;
            int *besty = &ybest;
            int *bestz = &zbest;
            int *bestt = &tbest;
            
            // PROPAGATION: Improve current guess by trying
            //instead correspondences from left and above
            //(below and right on odd iterations).
            
            //////////////////////////////////////////////////
            //X SHIFT
            //////////////////////////////////////////////////
            int pos_current_shift = pos_current - nn_change*ha;
            int bx = nnf[pos_current_shift] + nn_change;
            int by = nnf[pos_current_shift + size_a];
            int bz = nnf[pos_current_shift + size_ax2];
            
            int ind = by + bx*hb + bz*hb_wb;
            
            if (ind > 0 && (0 < mask[ind])) {
                
                int tp = nnf[pos_current_shift + size_ax3];
                int zp = bz+(tp*da);
                int pos_b = by + bz*hb + zp*hb*wb;
                
                d_tot = ssd_3D(subject, library, ha, wa, hb, wb, ay, ax, az, by, bx, zp, pw);
                
                //mutex *bestd
                if (d_tot < *bestd) {
                    
                    //mutex table
                    int ok = 0, ok2 = 0;
                    for (ok = 0; ok<multiple_pm_nbr; ok++) {
                        if (use_mutex)
                            sem_wait(&(mutex_list[pos_current_a + ok*size_a]));
                        if (pos_b == nnf_list_pos[pos_current_a + ok*size_a])
                            break;
                    }
                    if (ok == multiple_pm_nbr) { // not used in the list
                        *bestd = d_tot;
                        *bestx = bx;
                        *besty = by;
                        *bestz = bz;
                        *bestt = tp;
                        nnf_list_pos[pos_current_a + proc_nbr*size_a] = pos_b;
                    }
                    if (use_mutex) {
                        for (ok2 = 0; ok2<=MIN(ok,multiple_pm_nbr-1); ok2++) //free the mutex
                            sem_post(&(mutex_list[pos_current_a + ok2*size_a]));
                    }
                    
                }
            }
            
            //////////////////////////////////////////////////
            //Y SHIFT
            //////////////////////////////////////////////////
            pos_current_shift = pos_current -nn_change;
            bx = nnf[pos_current_shift];
            by = nnf[pos_current_shift + size_a] + nn_change;
            bz = nnf[pos_current_shift + size_ax2];
            
            ind = by + bx*hb +bz*hb_wb;
            
            if (ind > 0 && (0 < mask[ind])) {
                
                int tp = nnf[pos_current_shift + size_ax3];
                int zp = bz+(tp*da);
                int pos_b = by + bz*hb + zp*hb*wb;
                
                
                
                d_tot = ssd_3D(subject, library, ha, wa, hb, wb, ay, ax, az, by, bx, zp, pw);
                
                if (d_tot < *bestd) {
                    //mutex table
                    int ok = 0, ok2 = 0;
                    for (ok = 0; ok<multiple_pm_nbr; ok++) {
                        if (use_mutex)
                            sem_wait(&(mutex_list[pos_current_a + ok*size_a]));
                        if (pos_b == nnf_list_pos[pos_current_a + ok*size_a])
                            break;
                    }
                    if (ok == multiple_pm_nbr) { // not used in the list
                        *bestd = d_tot;
                        *bestx = bx;
                        *besty = by;
                        *bestz = bz;
                        *bestt = tp;
                        nnf_list_pos[pos_current_a + proc_nbr*size_a] = pos_b;
                    }
                    if (use_mutex) {
                        for (ok2 = 0; ok2<=MIN(ok,multiple_pm_nbr-1); ok2++) //free the mutex
                            sem_post(&(mutex_list[pos_current_a + ok2*size_a]));
                    }
                }
                
                
            }
            //////////////////////////////////////////////////
            //Z SHIFT
            //////////////////////////////////////////////////
            pos_current_shift = pos_current -nn_change*ha_wa;
            bx = nnf[pos_current_shift];
            by = nnf[pos_current_shift + size_a];
            bz = nnf[pos_current_shift + size_ax2] + nn_change;
            
            ind = by + bx*hb +bz*hb_wb;
            
            if (ind > 0 && (0 < mask[ind])) {
                
                
                int tp = (int) nnf[pos_current_shift + size_ax3];
                int zp = bz+(tp*da);
                int pos_b = by + bz*hb + zp*hb*wb;
                
                
                d_tot = ssd_3D(subject, library, ha, wa, hb, wb, ay, ax, az, by, bx, zp, pw);
                
                if (d_tot < *bestd) {
                    //mutex table
                    int ok = 0, ok2 = 0;
                    for (ok = 0; ok<multiple_pm_nbr; ok++) {
                        if (use_mutex)
                            sem_wait(&(mutex_list[pos_current_a + ok*size_a]));
                        if (pos_b == nnf_list_pos[pos_current_a + ok*size_a])
                            break;
                    }
                    if (ok == multiple_pm_nbr) { // not used in the list
                        *bestd = d_tot;
                        *bestx = bx;
                        *besty = by;
                        *bestz = bz;
                        *bestt = tp;
                        nnf_list_pos[pos_current_a + proc_nbr*size_a] = pos_b;
                    }
                    if (use_mutex) {
                        for (ok2 = 0; ok2<=MIN(ok,multiple_pm_nbr-1); ok2++) //free the mutex
                            sem_post(&(mutex_list[pos_current_a + ok2*size_a]));
                    }
                }
                
                
            }
            
            ////////////////////////////////////////////////////////
            //RANDOM SEARCH : fixed template
            //////////////////////////////////////////////////////
            int rs_start = 10; //mag_init;
            
            int xfix  = xbest;
            int yfix  = ybest;
            int zfix  = zbest;
            int tfix  = tbest;
            
            // Sampling window
            for (int mag = rs_start; mag >= 1; mag /= 2) {
                
                /* Box limits */
                int xmin = MAX(xfix-mag, pw);
                int xmax = MIN(xfix+mag+1,wa-pw);
                if(xmin == xmax) continue;
                
                int ymin = MAX(yfix-mag, pw);
                int ymax = MIN(yfix+mag+1,ha-pw);
                if(ymin == ymax) continue;
                
                int zmin = MAX(zfix-mag, pw);
                int zmax = MIN(zfix+mag+1,da-pw);
                if(zmin == zmax) continue;
                
                int rand_val = opal_rand(&next);
                int bx = xmin+rand_val%(xmax-xmin);
                rand_val = opal_rand(&next);
                int by = ymin+rand_val%(ymax-ymin);
                rand_val = opal_rand(&next);
                int bz = zmin+rand_val%(zmax-zmin);
                int zp = bz+tfix*da;
                
                int pos_b = by + bz*hb + zp*hb*wb;
                
                /* constrain the template */
                if (mask[by + bx*ha + bz*ha_wa] > 0) {
                    
                    dist = ssd_3D(subject, library, ha, wa, hb, wb, ay, ax, az, by, bx, zp, pw);
                    
                    if (dist < *bestd) {
                        //mutex table
                        int ok = 0, ok2 = 0;
                        for (ok = 0; ok<multiple_pm_nbr; ok++) {
                            if (use_mutex)
                                sem_wait(&(mutex_list[pos_current_a + ok*size_a]));
                            if (pos_b == nnf_list_pos[pos_current_a + ok*size_a])
                                break;
                        }
                        if (ok == multiple_pm_nbr) { // not used in the list
                            *bestd = d_tot;
                            *bestx = bx;
                            *besty = by;
                            *bestz = bz;
                            nnf_list_pos[pos_current_a + proc_nbr*size_a] = pos_b;
                        }
                        if (use_mutex) {
                            for (ok2 = 0; ok2<=MIN(ok,multiple_pm_nbr-1); ok2++) //free the mutex
                                sem_post(&(mutex_list[pos_current_a + ok2*size_a]));
                        }
                    }
                }
                
            }
            
            //Map updates
            nnf[pos_current] = *bestx;
            nnf[pos_current + size_a] = *besty;
            nnf[pos_current + size_ax2] = *bestz;
            nnf[pos_current + size_ax3] = *bestt;
            nnfd[pos_current_nnfd] = *bestd;
            
        }
        
        count++;
    }
    
    
    pthread_exit(0);
}




//////////////////////////////////////////////////////////////////////////
/////////////////////////////////// MAIN /////////////////////////////////
//////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)nlhs;
    (void)nrhs;
    
    /* INPUTS */
    //Load images: subject to segment and the library
    float* subject = (float*) mxGetPr(prhs[0]);
    float* library = (float*) mxGetPr(prhs[1]);
    
    //Dimensions stuff
    int subject_dims_nbr = mxGetNumberOfDimensions(prhs[0]);
    int library_dims_nbr = mxGetNumberOfDimensions(prhs[1]);
    if (subject_dims_nbr != library_dims_nbr) 
    {
        printf("Size Error: subject & library types are different");
        return;
    }

    const mwSize* subject_dims = mxGetDimensions(prhs[0]);
    const mwSize* library_dims = mxGetDimensions(prhs[1]);
    int subject_h = subject_dims[0];
    int subject_w = subject_dims[1];
    int subject_d = subject_dims[2];
    int library_h = library_dims[0];
    int library_w = library_dims[1];
    int library_d = library_dims[2];
    
    int idx = 2;
    
    unsigned char* mask = (unsigned char*) mxGetPr(prhs[idx++]);
    int iter = mxGetScalar(prhs[idx++]);
    int multiple_pm_nbr = mxGetScalar(prhs[idx++]);
 

    // Init NNF and NNFD
    int subject_size = subject_h*subject_w*subject_d*multiple_pm_nbr;
    
    mwSize dims2[5];
    dims2[0] = subject_h;
    dims2[1] = subject_w;
    dims2[2] = subject_d;
    dims2[3] = 4;
    dims2[4] = multiple_pm_nbr;
    plhs[0] = mxCreateNumericArray(5, dims2, mxINT32_CLASS, mxREAL);
    enc_type * nnf = (enc_type*)mxGetPr(plhs[0]);
    for (int i=0; i<4*subject_size; i++)
    {
        nnf[i] = 5;
    }

    dims2[3] = multiple_pm_nbr;
    plhs[1] = mxCreateNumericArray(4, dims2, mxSINGLE_CLASS, mxREAL);
    float * nnfd = (float*)mxGetPr(plhs[1]);

    // NEW STRUCTURES //
    int* nnf_list_pos = (int*) calloc(subject_size, sizeof(int)); // To contain the position of
    
    sem_t* mutex_list = (sem_t*) malloc(subject_size*sizeof(sem_t));
    for (int i=0;i<subject_size;i++)   
    {
        sem_init(&(mutex_list[i]), 0, 1);
    }
    
    int patch_radius = mxGetScalar(prhs[idx++]);
    int mag_init = mxGetScalar(prhs[idx++]);
    
    unsigned char *mask_path = (unsigned char *) mxGetPr(prhs[idx++]);
    int nb_vox = mxGetScalar(prhs[idx++]);
    
    //unsigned char *mask_seg = (unsigned char *) mxGetPr(prhs[idx++]);
    
    int use_mutex = (int) mxGetScalar(prhs[idx++]);
    
    //Thread argument structures
    pthread_t*   thread_list = (pthread_t*) calloc(multiple_pm_nbr, sizeof(pthread_t));
    opal_struct* thread_args = (opal_struct*)calloc(multiple_pm_nbr, sizeof(opal_struct));
    
    srand(time(NULL));
    unsigned long next = 1789;
   


//    //Launching of the THREADS
     for (int k=0;  k<2; k++)
    {
        int iter_k = iter;
        if (k==0) //just initialization
        {
            iter_k = 0;
        }

        for (int i=0; i<multiple_pm_nbr; i++) 
        {
            //Thread arguments
            thread_args[i].mask = mask;
            thread_args[i].mask_path = mask_path;
            thread_args[i].nb_vox = nb_vox;
            thread_args[i].wa = subject_w;
            thread_args[i].ha = subject_h;
            thread_args[i].da = subject_d;
            thread_args[i].wb = library_w;
            thread_args[i].hb = library_h;
            thread_args[i].db = library_d;
            thread_args[i].iter = iter_k;
            thread_args[i].proc_nbr = i % multiple_pm_nbr;
            thread_args[i].mag_init = mag_init;
            thread_args[i].patch_radius = patch_radius;
            thread_args[i].offset_mp = subject_w*subject_h*subject_d*(i % multiple_pm_nbr);
            //*knn_nbr;
            
            thread_args[i].nnf_list_pos = nnf_list_pos;
            thread_args[i].mutex_list = mutex_list;
            thread_args[i].multiple_pm_nbr = multiple_pm_nbr;
            thread_args[i].use_mutex = use_mutex;
            
            next = opal_rand(&next);
            thread_args[i].next = next;
            
            if (i<multiple_pm_nbr) 
            {
                thread_args[i].subject = subject;
                thread_args[i].library = library;
                thread_args[i].nnf = nnf;
                thread_args[i].nnfd = nnfd;
            }
            
            if (pthread_create(&thread_list[i], NULL, opal, &thread_args[i]))
                printf("Error creating a thread!\n");
            
        }
        
//        if (k==0) 
//        {
            //Wait for all PM threads to end
            for (int i=0; i<multiple_pm_nbr; i++) 
            {
                pthread_join(thread_list[i],NULL);
            }
//        }
        
    }
    
    //Precomputations
    //int ha = subject_h;
    //int wa = subject_w;
    //int da = subject_d;
    //int hb = library_h;
    //int wb = library_w;
    //int ha_wa = ha*wa;
    //int size_a = ha_wa*da;
    //int minslice,maxslice,nslices,Nthreads;
    //int ini, fin,k,i;
    // for (int i=0;i<size_a*h_list_lab;i++){
    //     sem_destroy(&mutex[i]);
    // }
    //free(mutex);
    for (int i=0;i<subject_size;i++)
    {
        sem_destroy(&mutex_list[i]);
    }
    free(mutex_list);
    free(nnf_list_pos);
}


