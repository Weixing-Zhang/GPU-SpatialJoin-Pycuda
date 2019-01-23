"""
Author: 
    Weixing Zhang
Date:   
    2018
Name:
    Spatial joint general functions
"""
import shapefile
import numpy as np
import itertools

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import driver, compiler, gpuarray, tools
from pycuda.compiler import SourceModule


def points2edges(pnt_array):
    """ Convert points array to edges array for further computation
    Explanation:
        For example of polygon points (three edges):
            pnts = np.array([[-72.16702464618417, 41.75743927963142], 
                             [-72.15220856950636, 41.75636293931843], 
                             [-72.16976706215449, 41.75673842800947], 
                             [-72.16702464618417, 41.75743927963142]])
    
            return will be:
                [[-72.16702465  41.75743928 -72.15220857  41.75636294]
                 [-72.15220857  41.75636294 -72.16976706  41.75673843]
                 [-72.16976706  41.75673843 -72.16702465  41.75743928]]
    """
    return np.hstack((pnt_array[:-1],pnt_array[1:]))
            
            
def read_shp(shp_path):
    """ read shape file
        read .shp file based on its shape file type as return bounding box as array
        LOWER LEFT (x,y) coordinate + UPPER RIGHT (x,y) coordinate
    
    Explanation: In this reading shape file process, the function can 
                 identify the type of shapefile and create
                 a output array as needed
    
    Parameters
    ----------
    shp_path: path to the shape file
    
    Returns
    ----------  
    sf_shapes: ready for reading information of the shapefile
    sf_array: array that converted from shapefile for filtering process
    """
    sf = shapefile.Reader(shp_path)
    sf_type_id = sf.shapeType
    
    if sf_type_id == 1 or sf_type_id == 8 or sf_type_id == 11:
        sf_type = 'POINT'
        print "Reading POINT shapefile: ", shp_path
    
    elif sf_type_id == 3 or sf_type_id == 13:
        sf_type = 'LINE'
        print "Reading LINE shapefile: ", shp_path
    
    elif sf_type_id == 5 or sf_type_id == 15:
        sf_type = 'POLYGON'
        print "Reading POLYGON shapefile: ", shp_path
     
    sf_shapes = sf.shapes()  
    
    # for point shape file
    if sf_type == "POINT":
        
        # create an empty output array
        sf_array =  np.zeros((len(sf_shapes),2))
        
        # go through each features
        for sf_id in xrange(len(sf_shapes)):
            
            # get the points of each feature
            sf_feature = sf_shapes[sf_id].points 
            sf_array[sf_id] = sf_feature[0]
    
    # for polygon shape file        
    elif sf_type == "POLYGON" or sf_type == "LINE":
        
        # create an empty output array
        sf_array =  np.zeros((len(sf_shapes),4))
        
        # go through each features
        for sf_id in xrange(len(sf_shapes)):
            
            #  lower left (x,y) coordinate + upper right (x,y) coordinate
            sf_feature_bbox = sf_shapes[sf_id].bbox
            sf_array[sf_id] = sf_feature_bbox
            
    return sf_shapes, sf_array


def GPU_serial_sorted_MBR_filter(shp_A_array,shp_B_array):
    """ implementation of sorted MBR filter on GPU device 
    
    It is developed based on serial_sorted_MBR_filter(shp_A_array,shp_B_array):
    
    NOTE: THIS ONE IS BETTER THAN improved_GPU_serial_sorted_MBR_filter and 
          loadbala_GPU_serial_sorted_MBR_filter
          
    Parameters
    ----------
    shp_A_array: the [bbox2_x1,bbox2_y1,bbox2_x2,bbox2_y2] for each bounding box 
                 for each polygon in polygon shapefile A
                 lower left (x,y) coordinate + upper right (x,y) coordinate
    shp_B_array: the [bbox2_x1,bbox2_y1,bbox2_x2,bbox2_y2] for each bounding box 
                 for each polygon in polygon shapefile B
    
    Returns
    ----------  
    final_MBR_list: [[i1_A,i1_B],...] the index for polygons from shapefile A and B whose
                    MBRs are intersected
    """
    
    # KERNEL FUNCTION
    # Please Note: In C programming: Maths operation only happen in same data type
    kernel_code = """    // line 0
    #include <stdio.h>   // line 1
    #include <math.h>    // line 2 ...
     
    __global__ void sorted_MBR_filter(int *sort_x_index, int *rank_x_index, \
                                      float *shp_A_y_array, float *shp_B_y_array, \
                                      int *interval_x_array, int *interval_x_index, \
                                      int *output_array, \
                                      int shp_A_num, int shp_B_num)
    {
        int threadId = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
        int start = threadId*2;
        int end = start+1;
        int index_start = 0;
        int i_X;
        int inter_count = 0;
        int possible_shp_A_index;
        int possible_shp_B_index;
        float y_i_1,y_i_2;
        float y_j_1,y_j_2;
        
        if (threadId == 0)
        {
            index_start; 
        }
        
        else if (threadId>=shp_A_num+shp_B_num)
        {
            return;
        }

        else 
        {
            index_start = interval_x_index[threadId-1]; 
        }
        
        // Implement the sorted MBR algorithm proposed by Aghajarian
        for (i_X=rank_x_index[start]+1;i_X<rank_x_index[end];i_X++) 
        {   
            if (start<shp_A_num*2 && sort_x_index[i_X]>=shp_A_num*2) 
            {
                possible_shp_A_index = start/2;
                possible_shp_B_index = (sort_x_index[i_X]-shp_A_num*2)/2;
            }
            
            else if (start>=shp_A_num*2 && sort_x_index[i_X]<shp_A_num*2)
            {
                possible_shp_A_index = sort_x_index[i_X]/2;
                possible_shp_B_index = (start-shp_A_num*2)/2;
            }
            
            else
            {
                continue;
            }
            
            // Make sure pair that overlap on x-axis overlap on y-axis as well
            y_i_1 = shp_A_y_array[possible_shp_A_index*2];
            y_i_2 = shp_A_y_array[possible_shp_A_index*2+1];
            y_j_1 = shp_B_y_array[possible_shp_B_index*2];
            y_j_2 = shp_B_y_array[possible_shp_B_index*2+1];
            
            // check if they intersect
            if (output_array[index_start*2+inter_count*2] == -1 && 
                output_array[index_start*2+inter_count*2+1] == -1)
            {
                if (y_i_2>=y_j_1 && y_j_2>=y_i_1)
                {
                    output_array[index_start*2+inter_count*2] = possible_shp_A_index;
                    output_array[index_start*2+inter_count*2+1] = possible_shp_B_index;
                }
            }
            
            // This is used to find out where to put the index 
            // of intersected polygons from shapefile A and B
            inter_count += 1;
        }
     }
    """
    
    # easy compromise
    shp_A_num = shp_A_array.shape[0]
    shp_B_num = shp_B_array.shape[0]

    # X coordinate
    # read all X1,X2 from polygon A + all X1,X2 from polygon B
    x_array = np.concatenate((np.dstack((shp_A_array[:,0],shp_A_array[:,2])).flatten(),
                              np.dstack((shp_B_array[:,0],shp_B_array[:,2])).flatten()),
                              axis=0)
    
    # sorted index and rank index of sorted X
    sort_x_index = np.argsort(x_array).astype(np.int32)
    rank_x_index = np.argsort(sort_x_index).astype(np.int32)
    
    # for later y overlap
    shp_A_y_array = np.dstack((shp_A_array[:,1],shp_A_array[:,3])).flatten()
    shp_B_y_array = np.dstack((shp_B_array[:,1],shp_B_array[:,3])).flatten()
    shp_A_y_array = shp_A_y_array.astype(np.float32)
    shp_B_y_array = shp_B_y_array.astype(np.float32)

    # ---------------- Prepare for GPU ---------------- 
    # get the number for predefine array
    interval_x_array = np.zeros((shp_A_num+shp_B_num),np.int32)
    
    # each Xj,0 and Xj,1 
    for j in xrange(0,(shp_A_num+shp_B_num)*2,2):
        
        # index of Xi,0
        start = j
        # index of Xi,1
        end = start+1       
        
        interval_x_array[j/2] = rank_x_index[end]-rank_x_index[start]-1

    # compute the X coordinate first then Y 
    interval_x_index = np.add.accumulate(interval_x_array)    
    count_x_sum = np.sum(interval_x_array)
    # ------------------------------------------------- 
    
    # Transfer data to GPU
    # INPUT 1: the sorted index for sorted X
    sort_x_index_GPU = gpuarray.to_gpu(sort_x_index)
    # INPUT 2: the rank index for the sorted index of sorted X
    rank_x_index_GPU = gpuarray.to_gpu(rank_x_index)
    # INPUT 3: for y value overlapping
    shp_A_y_array_GPU = gpuarray.to_gpu(shp_A_y_array)
    shp_B_y_array_GPU = gpuarray.to_gpu(shp_B_y_array)
    # INPUT 4: shared index for GPU to check which pair they are working on
    interval_x_array = interval_x_array.astype(np.int32)
    interval_x_array_GPU = gpuarray.to_gpu(interval_x_array)
    interval_x_index = interval_x_index.astype(np.int32)
    interval_x_index_GPU = gpuarray.to_gpu(interval_x_index)

    # OUTPUT 1: list for all intersected pairs
    output_array = np.zeros((count_x_sum*2))
    output_array[:] = -1
    output_array = output_array.astype(np.int32)
    output_array_GPU = gpuarray.to_gpu(output_array)
    
    # make sure enough threads are deployed
    max_block_size = 512
    if shp_A_num+shp_B_num<=max_block_size:
        max_block_size = shp_A_num+shp_B_num
        max_grid_size = 1
    else:
        max_grid_size = ((shp_A_num+shp_B_num)/max_block_size)+1

    # get the kernel function
    mod = compiler.SourceModule(kernel_code)
    func_count = mod.get_function("sorted_MBR_filter")
    
    func_count(
         # input
         sort_x_index_GPU, rank_x_index_GPU, 
         shp_A_y_array_GPU, shp_B_y_array_GPU,
         interval_x_array_GPU, interval_x_index_GPU,
         # output
         output_array_GPU, 
         # some fixed variables
         np.int32(shp_A_num), np.int32(shp_B_num),
         # block size
         block=(max_block_size,1,1), 
         # gird size
         grid=(max_grid_size,1,1))
 
    final_output = output_array_GPU.get()
    final_output = final_output[final_output>=0] # The defined output array set -1 for unintersected pairs

    # remove duplication in return
    final_output = (np.unique(np.reshape(final_output, (final_output.shape[0]/2,2)),axis=0)).tolist()

    return final_output


def GPU_common_MBR_filter(shp_A_array,shp_B_array,
                          shp_A,shp_B,
                          filter_index_xy):
    """ Conduct GPU common MBR filter
    
    This filter tries to categorize the result from the first filter into 3 types:
    (1) within, one MBR contains anther MBR
    (2) edge intersect, edge(s) of one polygon intersect the intersected MBR
    (3) disjoint, which should be discarded
    
    Parameters
    ----------
    shp_A_array: the [bbox2_x1,bbox2_y1,bbox2_x2,bbox2_y2] for each bounding box 
                 for each polygon in polygon shapefile A
    shp_B_array: the [bbox2_x1,bbox2_y1,bbox2_x2,bbox2_y2] for each bounding box 
                 for each polygon in polygon shapefile B
    shp_A,shp_B: the read shapefile of A and B with information about each vertice
    filter_index_xy: the result from first filter, whose MBRs are intersected
    
    Returns
    ----------  
    W_list: for pairs in W list, they will be tested with "point in polygon";
            [A,B]: means Polygon A within polygon B
    I_list: for pairs in I list, they will be tested with simple intersection 
            and save potential edges for each candidate polygon for final "edge intersection"
    plyn_A_poten_edges_dict: tells which edge from polygon from shapefile A needs to be checked
                             in this way, lots of edges do not need to check at all
    plyn_B_poten_edges_dict: tells which edge from polygon from shapefile B needs to be checked
    """
    # KERNEL FUNCTION
    # Please Note: In C programming: Maths operation only happen in same data type
    kernel_code = """    // line 0
    #include <stdio.h>   // line 1
    #include <math.h>    // line 2 ...
     
    __global__ void sorted_MBR_filter(int *filter_index_xy, \
                                      float *shp_A_array, float *shp_B_array, \
                                      float *plyn_A_edges, float *plyn_B_edges, \
                                      int *plyn_A_edges_index, int *plyn_B_edges_index, \
                                      int *W_output_array, int *I_output_array, \
                                      int *E_plyn_A_array, int *E_plyn_B_array, \
                                      int *E_plyn_A_shadow_array, int *E_plyn_B_shadow_array, \
                                      int len_filter_index_xy)
    {
        int threadId = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
        int polygon_A_start, polygon_A_end;
        int polygon_B_start, polygon_B_end;
        
        float A_X, A_Y, B_X, B_Y; // easier for line intersection
        float C_X, C_Y, D_X, D_Y; // easier for line intersection
        int ccw_ACD, ccw_BCD, ccw_ABC, ccw_ABD; // for line intersection
        int intersect_A_index, intersect_B_index;
        int intersect_A_ctr, intersect_B_ctr;
        
        if (threadId>=len_filter_index_xy) return;
        
        // get plyn id
        int plyn_A_id = filter_index_xy[threadId*2];
        int plyn_B_id = filter_index_xy[threadId*2+1];
        
        // get four corners of MBR of polygon A
        float plyn_A_MBR_0 = shp_A_array[plyn_A_id*4];
        float plyn_A_MBR_1 = shp_A_array[plyn_A_id*4+1];
        float plyn_A_MBR_2 = shp_A_array[plyn_A_id*4+2];
        float plyn_A_MBR_3 = shp_A_array[plyn_A_id*4+3];
        
        // get four corners of MBR of polygon B
        float plyn_B_MBR_0 = shp_B_array[plyn_B_id*4];
        float plyn_B_MBR_1 = shp_B_array[plyn_B_id*4+1];
        float plyn_B_MBR_2 = shp_B_array[plyn_B_id*4+2];
        float plyn_B_MBR_3 = shp_B_array[plyn_B_id*4+3];
        
        // find two Max and two Min
        float intersct_xA = (plyn_A_MBR_0>plyn_B_MBR_0)?(plyn_A_MBR_0):(plyn_B_MBR_0);
        float intersct_yA = (plyn_A_MBR_1>plyn_B_MBR_1)?(plyn_A_MBR_1):(plyn_B_MBR_1);
        float intersct_xB = (plyn_A_MBR_2<plyn_B_MBR_2)?(plyn_A_MBR_2):(plyn_B_MBR_2);
        float intersct_yB = (plyn_A_MBR_3<plyn_B_MBR_3)?(plyn_A_MBR_3):(plyn_B_MBR_3);

        // compare
        if ((plyn_A_MBR_0==intersct_xA) && 
            (plyn_A_MBR_1==intersct_yA) && 
            (plyn_A_MBR_2==intersct_xB) && 
            (plyn_A_MBR_3==intersct_yB))
            {   
                // "Polygon A ", plyn_A_id, " within polygon B ", plyn_B_id
                W_output_array[threadId*2] = plyn_A_id;
                W_output_array[threadId*2+1] = plyn_B_id;
            }
            
        else if ((plyn_B_MBR_0==intersct_xA) && 
                 (plyn_B_MBR_1==intersct_yA) && 
                 (plyn_B_MBR_2==intersct_xB) && 
                 (plyn_B_MBR_3==intersct_yB))
            {   
                // "Polygon B ", plyn_B_id, " within polygon A ", plyn_A_id
                W_output_array[threadId*2] = plyn_A_id;
                W_output_array[threadId*2+1] = plyn_B_id;
            }
        
        else
            {
                // Construct Intersect MBR vertices (from starting point to starting point)
                float intersect_MBR_vtices[10];
                intersect_MBR_vtices[0] = intersct_xA;
                intersect_MBR_vtices[1] = intersct_yA;
                intersect_MBR_vtices[2] = intersct_xA;
                intersect_MBR_vtices[3] = intersct_yB;
                intersect_MBR_vtices[4] = intersct_xB;
                intersect_MBR_vtices[5] = intersct_yB;
                intersect_MBR_vtices[6] = intersct_xB;
                intersect_MBR_vtices[7] = intersct_yA;
                intersect_MBR_vtices[8] = intersct_xA;
                intersect_MBR_vtices[9] = intersct_yA;
     
                // prepare edges for MBR
                int edge_i; // counter for MBR edge
                int edges_A; // counter for edges of the polygon A
                int edges_B; // counter for edges of the polygon B
                
                // initialize counter
                intersect_A_ctr = 0;
                intersect_B_ctr = 0;
                
                for (edge_i=0;edge_i<4;edge_i++)
                {
                    A_X = intersect_MBR_vtices[edge_i*2];    // That's A; MBR_edge_start_X
                    A_Y = intersect_MBR_vtices[edge_i*2+1];  //           MBR_edge_start_Y
                    B_X = intersect_MBR_vtices[edge_i*2+2];  // That's B; MBR_edge_end_X
                    B_Y = intersect_MBR_vtices[edge_i*2+3];  //           MBR_edge_end_Y
                    
                    // find the index range
                    if (threadId==0) 
                    {
                        polygon_A_start = 0;
                        polygon_B_start = 0;
                        polygon_A_end = plyn_A_edges_index[0];
                        polygon_B_end = plyn_B_edges_index[0];
                    }
                    else
                    {
                        polygon_A_start = plyn_A_edges_index[threadId-1];
                        polygon_B_start = plyn_B_edges_index[threadId-1];
                        polygon_A_end = plyn_A_edges_index[threadId];
                        polygon_B_end = plyn_B_edges_index[threadId];
                    }
                    
                    // loop through edges for Polygon A
                    for (edges_A=polygon_A_start; edges_A<polygon_A_end;edges_A++)
                    {
                        intersect_A_index = 0; // Initialize intersect index
                        C_X = plyn_A_edges[edges_A*4];   // That's C; polygon_A_edge_start_X
                        C_Y = plyn_A_edges[edges_A*4+1]; //           polygon_A_edge_start_Y
                        D_X = plyn_A_edges[edges_A*4+2]; // That's D; polygon_A_edge_end_X
                        D_Y = plyn_A_edges[edges_A*4+3]; //           polygon_A_edge_end_Y
                        
                        // implement the ccw and intersect function; eg: total = (total<2500)?(1):(10);
                        ccw_ACD = ( ( (D_Y-A_Y)*(C_X-A_X) ) > ( (C_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                        ccw_BCD = ( ( (D_Y-B_Y)*(C_X-B_X) ) > ( (C_Y-B_Y)*(D_X-B_X) ) )?(1):(0);
                        ccw_ABC = ( ( (C_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(C_X-A_X) ) )?(1):(0);
                        ccw_ABD = ( ( (D_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                        
                        // check if they are intersected
                        intersect_A_index = ((ccw_ACD!=ccw_BCD)&&(ccw_ABC!=ccw_ABD))?(1):(0);

                        if (intersect_A_index)
                        {
                            intersect_A_ctr++;
                            E_plyn_A_shadow_array[edges_A] = 1;
                        }
                        
                    }
                    
                    // loop through edges for Polygon B
                    for (edges_B=polygon_B_start; edges_B<polygon_B_end;edges_B++)
                    {
                        intersect_B_index = 0;           // Initialize intersect index
                        C_X = plyn_B_edges[edges_B*4];   // That's C; polygon_B_edge_start_X
                        C_Y = plyn_B_edges[edges_B*4+1]; //           polygon_B_edge_start_Y
                        D_X = plyn_B_edges[edges_B*4+2]; // That's D; polygon_B_edge_end_X
                        D_Y = plyn_B_edges[edges_B*4+3]; //           polygon_B_edge_end_Y
                        
                        // implement the ccw and intersect function; eg: total = (total<2500)?(1):(10);
                        ccw_ACD = ( ( (D_Y-A_Y)*(C_X-A_X) ) > ( (C_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                        ccw_BCD = ( ( (D_Y-B_Y)*(C_X-B_X) ) > ( (C_Y-B_Y)*(D_X-B_X) ) )?(1):(0);
                        ccw_ABC = ( ( (C_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(C_X-A_X) ) )?(1):(0);
                        ccw_ABD = ( ( (D_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                        
                        // check if they are intersected
                        intersect_B_index = ((ccw_ACD!=ccw_BCD)&&(ccw_ABC!=ccw_ABD))?(1):(0);
                        
                        if (intersect_B_index)
                        {
                            intersect_B_ctr++;
                            E_plyn_B_shadow_array[edges_B] = 1;
                        }
                    }
                }
                
                // Check the condition
                if (intersect_A_ctr==0 || intersect_B_ctr==0)
                {
                    return;
                }
                
                else
                {
                    I_output_array[threadId*2] = plyn_A_id;
                    I_output_array[threadId*2+1] = plyn_B_id;
                    
                    // find the index range
                    if (threadId==0) 
                    {
                        polygon_A_start = 0;
                        polygon_B_start = 0;
                        polygon_A_end = plyn_A_edges_index[0];
                        polygon_B_end = plyn_B_edges_index[0];
                    }
                    else
                    {
                        polygon_A_start = plyn_A_edges_index[threadId-1];
                        polygon_B_start = plyn_B_edges_index[threadId-1];
                        polygon_A_end = plyn_A_edges_index[threadId];
                        polygon_B_end = plyn_B_edges_index[threadId];
                    }
                    
                    for (edges_A=polygon_A_start; edges_A<polygon_A_end; edges_A++)
                    {
                        E_plyn_A_array[edges_A] = E_plyn_A_shadow_array[edges_A];
                    }
                    
                    for (edges_B=polygon_B_start; edges_B<polygon_B_end; edges_B++)
                    {
                        E_plyn_B_array[edges_B] = E_plyn_B_shadow_array[edges_B];
                    } 
                }
            }     
     }
    """ 
    # INPUT 1:
    filter_index_xy = np.asarray(filter_index_xy, np.int32)
    filter_index_xy_GPU = gpuarray.to_gpu(filter_index_xy)
    # INPUT 2:
    shp_A_array = shp_A_array.astype(np.float32)
    shp_A_array_GPU = gpuarray.to_gpu(shp_A_array)
    # INPUT 3:
    shp_B_array = shp_B_array.astype(np.float32)
    shp_B_array_GPU = gpuarray.to_gpu(shp_B_array)
     
    """ Convert vertices of each polygon into array
    """
    plyn_A_edges_index = list()
    plyn_B_edges_index = list() 
    plyn_A_edges = list()
    plyn_B_edges = list()
    
    for plyn_A_id,plyn_B_id in filter_index_xy:
        plyn_A_edges.append(points2edges(shp_A[plyn_A_id].points))
        plyn_A_edges_index.append(len(shp_A[plyn_A_id].points)-1) # The number of edge is 1 less than points
        plyn_B_edges.append(points2edges(shp_B[plyn_B_id].points))
        plyn_B_edges_index.append(len(shp_B[plyn_B_id].points)-1) # The number of edge is 1 less than points

    plyn_A_edges = list(itertools.chain.from_iterable(plyn_A_edges))
    plyn_B_edges = list(itertools.chain.from_iterable(plyn_B_edges))
    
    # convert vertices and their index into array for GPU 
    plyn_A_edges = np.asarray(plyn_A_edges)
    plyn_B_edges = np.asarray(plyn_B_edges)   
    plyn_A_edges_index = np.add.accumulate(np.asarray(plyn_A_edges_index))
    plyn_B_edges_index = np.add.accumulate(np.asarray(plyn_B_edges_index))
    
    # INPUT 4:
    plyn_A_edges = np.asarray(plyn_A_edges, np.float32)
    plyn_A_edges_GPU = gpuarray.to_gpu(plyn_A_edges)
    # INPUT 5:
    plyn_B_edges = np.asarray(plyn_B_edges, np.float32)
    plyn_B_edges_GPU = gpuarray.to_gpu(plyn_B_edges)
    # INPUT 6:
    plyn_A_edges_index = np.asarray(plyn_A_edges_index, np.int32)
    plyn_A_edges_index_GPU = gpuarray.to_gpu(plyn_A_edges_index)
    # INPUT 7:
    plyn_B_edges_index = np.asarray(plyn_B_edges_index, np.int32)
    plyn_B_edges_index_GPU = gpuarray.to_gpu(plyn_B_edges_index)
     
    # OUTPUT 1-1:
    W_output_array = np.zeros((len(filter_index_xy)*2))
    W_output_array[:] = -1
    W_output_array = W_output_array.astype(np.int32)
    W_output_array_GPU = gpuarray.to_gpu(W_output_array)
    
    # OUTPUT 2:
    I_output_array = np.zeros((len(filter_index_xy)*2))
    I_output_array[:] = -1
    I_output_array = I_output_array.astype(np.int32)
    I_output_array_GPU = gpuarray.to_gpu(I_output_array)
    
    # OUTPUT 3:
    E_plyn_A_array = np.zeros(plyn_A_edges_index[-1])
    E_plyn_A_array[:] = -1
    E_plyn_A_array = E_plyn_A_array.astype(np.int32)
    E_plyn_A_array_GPU = gpuarray.to_gpu(E_plyn_A_array)
    
    # OUTPUT 4:
    E_plyn_B_array = np.zeros(plyn_B_edges_index[-1])
    E_plyn_B_array[:] = -1
    E_plyn_B_array = E_plyn_B_array.astype(np.int32)
    E_plyn_B_array_GPU = gpuarray.to_gpu(E_plyn_B_array)
    
    # INTERNAL: 5:
    E_plyn_A_shadow_array_GPU = gpuarray.to_gpu(np.array(E_plyn_A_array))
    
    # INTERNAL 6:
    E_plyn_B_shadow_array_GPU = gpuarray.to_gpu(np.array(E_plyn_B_array))
    
    # Start using GPU
    max_block_size = 512
    if len(filter_index_xy)<=max_block_size:
        max_block_size = len(filter_index_xy)
        max_grid_size = 1
    else:
        max_grid_size = (len(filter_index_xy)/max_block_size)+1
    
    # get the kernel function
    mod = compiler.SourceModule(kernel_code)
    func_count = mod.get_function("sorted_MBR_filter")
    
    func_count(
         # input
         filter_index_xy_GPU, 
         shp_A_array_GPU, shp_B_array_GPU,
         plyn_A_edges_GPU, plyn_B_edges_GPU,
         plyn_A_edges_index_GPU, plyn_B_edges_index_GPU,
         # output
         W_output_array_GPU, I_output_array_GPU,
         E_plyn_A_array_GPU, E_plyn_B_array_GPU,
         E_plyn_A_shadow_array_GPU, E_plyn_B_shadow_array_GPU,
         # some fixed variables
         np.int32(len(filter_index_xy)), 
         # block size
         block=(max_block_size,1,1), 
         # gird size
         grid=(max_grid_size,1,1))
    
    # get W_list 
    W_output = W_output_array_GPU.get()
    W_output = W_output[W_output>=0]
    W_output = (np.unique(np.reshape(W_output, (W_output.shape[0]/2,2)),axis=0)).tolist()
    
    # get I_list
    I_output = I_output_array_GPU.get()
    I_output = I_output[I_output>=0]
    I_output = (np.unique(np.reshape(I_output, (I_output.shape[0]/2,2)),axis=0)).tolist()
    
    # get plyn_A_poten_edges
    plyn_A_poten_edges = E_plyn_A_array_GPU.get()
    plyn_A_poten_edges_dict = edge_accum2edge_index(plyn_A_poten_edges,plyn_A_edges_index, filter_index_xy, 0)
    
    # get plyn_A_poten_edges
    plyn_B_poten_edges = E_plyn_B_array_GPU.get()
    plyn_B_poten_edges_dict = edge_accum2edge_index(plyn_B_poten_edges,plyn_B_edges_index, filter_index_xy, 1)
    
    return W_output, I_output, plyn_A_poten_edges_dict, plyn_B_poten_edges_dict


def edge_accum2edge_index(plyn_edge_array,plyn_edge_index, filter_index_xy, plyn_name):
    """ Serve for GPU_common_MBR_filter function above
    
    Explanation: The output of GPU_common_MBR_filter includes candidate edge pairs that
                 tells which edge from polygon from shapefile A or B needs to be checked
                 in this way, lots of edges do not need to check at all
    
    Parameters
    ----------
    plyn_edge_array: the array of candidate edges
    plyn_edge_index: the array of index of candidate edges
    filter_index_xy: the result from first filter, whose MBRs are intersected
    plyn_name: 0 means the shapefile A; 1 means the shapfile B
    
    Returns
    ----------  
    output_dict: tells which edge from polygon from shapefile A needs to be checked
                 in this way, lots of edges do not need to check at all 
    """
    output_dict = {}
    counter = 0
    
    for pair_id, plyn_index_pair in enumerate(filter_index_xy):
        plyn_index = plyn_index_pair[plyn_name]
        
        added_index = 0
        dict_key = "%d-%d"%(counter,plyn_index)
        
        if pair_id == 0:
            start = 0
            end = plyn_edge_index[0]
            
            for i in xrange(start, end):
                if plyn_edge_array[i] == 1:
                    added_index = 1
                    if dict_key in output_dict and i-start not in output_dict[dict_key]:
                        output_dict[dict_key].append(i-start)
                    if dict_key not in output_dict:
                        output_dict[dict_key] = []
                        output_dict[dict_key].append(i-start)
                else:
                    continue
        
        else:
            start = plyn_edge_index[pair_id-1]
            end = plyn_edge_index[pair_id]
            
            for i in xrange(start, end):
                if plyn_edge_array[i] == 1:
                    added_index = 1
                    if dict_key in output_dict and i-start not in output_dict[dict_key]:
                        output_dict[dict_key].append(i-start)
                    if dict_key not in output_dict:
                        output_dict[dict_key] = []
                        output_dict[dict_key].append(i-start)
                else:
                    continue
        
        if added_index:
            counter +=1;
           
    return output_dict


def GPU_PnPtest(shp_A_array,shp_B_array,shp_A,shp_B,W_list):
    """ GPU Point-in-Polygon test for W list

    Explanation: check the result from common_MBR_filter for 
                 pair with "within relationship"
    
    Parameters
    ----------
    shp_A_array: the [bbox2_x1,bbox2_y1,bbox2_x2,bbox2_y2] for each bounding box 
                 for each polygon in polygon shapefile A
    shp_B_array: the [bbox2_x1,bbox2_y1,bbox2_x2,bbox2_y2] for each bounding box 
                 for each polygon in polygon shapefile B
    shp_A,shp_B: the read shapefile of A and B with information about each vertice
    W_list: pair with "within relationship" from common MBR filter result
    
    Returns
    ----------  
    W_intersect_result: for pairs in W list that are intersect
    """
    # KERNEL FUNCTION
    # Please Note: In C programming: Maths operation only happen in same data type
    kernel_code = """    // line 0
    #include <stdio.h>   // line 1
    #include <math.h>    // line 2 ...
     
    __global__ void PnPtest(int *W_array, \
                            float *shp_A_mbr, float *shp_B_mbr, \
                            float *shp_A_points, float *shp_B_points, \
                            int *shp_A_points_index, int *shp_B_points_index, \
                            float *shp_A_edges, float *shp_B_edges, \
                            int *shp_A_edges_index, int *shp_B_edges_index, \
                            int *W_intersect_array, int *Need_intersect_array, \
                            int len_W_list)
    {
        int threadId = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
        
        float A_X, A_Y, B_X, B_Y; // easier for line intersection
        float C_X, C_Y, D_X, D_Y; // easier for line intersection
        int ccw_ACD, ccw_BCD, ccw_ABC, ccw_ABD; // for line intersection
        int plyn_A_pnt_start, plyn_A_pnt_end;
        int plyn_B_pnt_start, plyn_B_pnt_end;
        int plyn_A_edge_start, plyn_A_edge_end;
        int plyn_B_edge_start, plyn_B_edge_end;
        int intersect_index;
        int count_left, count_right;
        
        // stop extra threads
        if (threadId>=len_W_list) return;
        
        /*
        int plyn_A_id = W_array[threadId*2];
        int plyn_B_id = W_array[threadId*2+1];
        printf("I am %d thread for A %d, B %d\\n", threadId, plyn_A_id, plyn_B_id);*/
        
        // read MBR of shp A and B
        float plyn_A_upperright_X = shp_A_mbr[threadId*4+2];
        float plyn_A_upperright_Y = shp_A_mbr[threadId*4+3];
        float plyn_B_upperright_X = shp_B_mbr[threadId*4+2];
        float plyn_B_upperright_Y = shp_B_mbr[threadId*4+3]; 
        
        // read this for ray distance
        float plyn_A_left_X = shp_A_mbr[threadId*4+0];
        float plyn_B_left_X = shp_B_mbr[threadId*4+0];
        
        // first check which polygon is within the other
        if  (plyn_A_upperright_X<=plyn_B_upperright_X && plyn_A_upperright_Y<=plyn_B_upperright_Y)
        {
            // A is within B -> test if any point of A within B
            // read Points from shp A AND read Edges from shp B
            
            // find the index range
            if (threadId==0) 
            {
                plyn_A_pnt_start = 0;
                plyn_A_pnt_end = shp_A_points_index[0];
                plyn_B_edge_start = 0;
                plyn_B_edge_end = shp_B_edges_index[0];
            }
            
            else
            {
                plyn_A_pnt_start = shp_A_points_index[threadId-1];
                plyn_A_pnt_end = shp_A_points_index[threadId];
                plyn_B_edge_start = shp_B_edges_index[threadId-1];
                plyn_B_edge_end = shp_B_edges_index[threadId];
            }
            
            
            /* Point in Polygon Test*/
            // get the bbox x axis range from shp B
            float mbr_x_range =  plyn_B_upperright_X-plyn_B_left_X;
            
            int plyn_A_pnt_index;
            int plyn_B_edge_index;
            for (plyn_A_pnt_index=plyn_A_pnt_start; plyn_A_pnt_index<plyn_A_pnt_end; plyn_A_pnt_index++)
            {
                // for testing if point in polygon or not
                count_left = 0;
                count_right = 0;
                
                // get x and y for each point
                float plyn_A_pnt_X = shp_A_points[plyn_A_pnt_index*2];
                float plyn_A_pnt_Y = shp_A_points[plyn_A_pnt_index*2+1];
                
                // go through each edge from shp B
                for (plyn_B_edge_index = plyn_B_edge_start; plyn_B_edge_index<plyn_B_edge_end; plyn_B_edge_index++)
                {
                    // create ray for right and left sides
                    float left_end_X = plyn_A_pnt_X - mbr_x_range;
                    float left_end_Y = plyn_A_pnt_Y;
                    float right_end_X = plyn_A_pnt_X + mbr_x_range;
                    float right_end_Y = plyn_A_pnt_Y;
                    
                    // test left first
                    A_X = plyn_A_pnt_X;
                    A_Y = plyn_A_pnt_Y;
                    B_X = left_end_X;
                    B_Y = left_end_Y;
                    C_X = shp_B_edges[plyn_B_edge_index*4];
                    C_Y = shp_B_edges[plyn_B_edge_index*4+1];
                    D_X = shp_B_edges[plyn_B_edge_index*4+2];
                    D_Y = shp_B_edges[plyn_B_edge_index*4+3];
                    
                    // implement the ccw and intersect function; eg: total = (total<2500)?(1):(10);
                    ccw_ACD = ( ( (D_Y-A_Y)*(C_X-A_X) ) > ( (C_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                    ccw_BCD = ( ( (D_Y-B_Y)*(C_X-B_X) ) > ( (C_Y-B_Y)*(D_X-B_X) ) )?(1):(0);
                    ccw_ABC = ( ( (C_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(C_X-A_X) ) )?(1):(0);
                    ccw_ABD = ( ( (D_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                    
                    // check if they are intersected
                    intersect_index = ((ccw_ACD!=ccw_BCD)&&(ccw_ABC!=ccw_ABD))?(1):(0);
                    
                    if (intersect_index) count_left++;
                    
                    // test right 
                    B_X = right_end_X;
                    B_Y = right_end_Y;
                    
                    // implement the ccw and intersect function; eg: total = (total<2500)?(1):(10);
                    ccw_ACD = ( ( (D_Y-A_Y)*(C_X-A_X) ) > ( (C_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                    ccw_BCD = ( ( (D_Y-B_Y)*(C_X-B_X) ) > ( (C_Y-B_Y)*(D_X-B_X) ) )?(1):(0);
                    ccw_ABC = ( ( (C_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(C_X-A_X) ) )?(1):(0);
                    ccw_ABD = ( ( (D_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                    
                    // check if they are intersected
                    intersect_index = ((ccw_ACD!=ccw_BCD)&&(ccw_ABC!=ccw_ABD))?(1):(0);
                    
                    if (intersect_index) count_right++;
                    
                }
                
                // printf("I am %d thread for A within B -- pnt A: X %f,  Y %f; has counter_left: %d; has counter_right: %d\\n", threadId, plyn_A_pnt_X, plyn_A_pnt_Y, count_left, count_right);
                
                if (count_left % 2 == 0 || count_right % 2 == 0) continue;
                else 
                {
                    W_intersect_array[threadId] = 1;
                    return;
                }
            }  
            
            if (W_intersect_array[threadId] == 0) Need_intersect_array[threadId] = 1;
        }
        
        else
        {
            // B is within A -> test if any point of B within A
            // read Points from shp B AND read Edges from shp A
            
            // find the index range
            if (threadId==0) 
            {
                plyn_B_pnt_start = 0;
                plyn_B_pnt_end = shp_B_points_index[0];
                plyn_A_edge_start = 0;
                plyn_A_edge_end = shp_A_edges_index[0];
            }
            
            else
            {
                plyn_B_pnt_start = shp_B_points_index[threadId-1];
                plyn_B_pnt_end = shp_B_points_index[threadId];
                plyn_A_edge_start = shp_A_edges_index[threadId-1];
                plyn_A_edge_end = shp_A_edges_index[threadId];
            }
            
            /* Point in Polygon Test*/
            // get the bbox x axis range from shp B
            float mbr_x_range =  plyn_A_upperright_X-plyn_A_left_X;
            
            int plyn_B_pnt_index;
            int plyn_A_edge_index;
            for (plyn_B_pnt_index=plyn_B_pnt_start; plyn_B_pnt_index<plyn_B_pnt_end; plyn_B_pnt_index++)
            {
                // for testing if point in polygon or not
                count_left = 0;
                count_right = 0;
                
                // get x and y for each point
                float plyn_B_pnt_X = shp_B_points[plyn_B_pnt_index*2];
                float plyn_B_pnt_Y = shp_B_points[plyn_B_pnt_index*2+1];
                    
                // go through each edge from shp B
                for (plyn_A_edge_index = plyn_A_edge_start; plyn_A_edge_index<plyn_A_edge_end; plyn_A_edge_index++)
                { 
                    // create ray for right and left sides
                    float left_end_X = plyn_B_pnt_X - mbr_x_range;
                    float left_end_Y = plyn_B_pnt_Y;
                    float right_end_X = plyn_B_pnt_X + mbr_x_range;
                    float right_end_Y = plyn_B_pnt_Y;
                    
                    // test left first
                    A_X = plyn_B_pnt_X;
                    A_Y = plyn_B_pnt_Y;
                    B_X = left_end_X;
                    B_Y = left_end_Y;
                    C_X = shp_A_edges[plyn_A_edge_index*4];
                    C_Y = shp_A_edges[plyn_A_edge_index*4+1];
                    D_X = shp_A_edges[plyn_A_edge_index*4+2];
                    D_Y = shp_A_edges[plyn_A_edge_index*4+3];
                    
                    // implement the ccw and intersect function; eg: total = (total<2500)?(1):(10);
                    ccw_ACD = ( ( (D_Y-A_Y)*(C_X-A_X) ) > ( (C_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                    ccw_BCD = ( ( (D_Y-B_Y)*(C_X-B_X) ) > ( (C_Y-B_Y)*(D_X-B_X) ) )?(1):(0);
                    ccw_ABC = ( ( (C_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(C_X-A_X) ) )?(1):(0);
                    ccw_ABD = ( ( (D_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                    
                    // check if they are intersected
                    intersect_index = ((ccw_ACD!=ccw_BCD)&&(ccw_ABC!=ccw_ABD))?(1):(0);
                    
                    if (intersect_index) count_left++;
                    
                    // test right 
                    B_X = right_end_X;
                    B_Y = right_end_Y;
                    
                    // implement the ccw and intersect function; eg: total = (total<2500)?(1):(10);
                    ccw_ACD = ( ( (D_Y-A_Y)*(C_X-A_X) ) > ( (C_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                    ccw_BCD = ( ( (D_Y-B_Y)*(C_X-B_X) ) > ( (C_Y-B_Y)*(D_X-B_X) ) )?(1):(0);
                    ccw_ABC = ( ( (C_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(C_X-A_X) ) )?(1):(0);
                    ccw_ABD = ( ( (D_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                    
                    // check if they are intersected
                    intersect_index = ((ccw_ACD!=ccw_BCD)&&(ccw_ABC!=ccw_ABD))?(1):(0);
                    
                    if (intersect_index) count_right++;
                    
                }
                
                // printf("I am %d thread for B within A -- pnt: X %f,  Y %f; has counter_left: %d; has counter_right: %d\\n", threadId, plyn_B_pnt_X, plyn_B_pnt_Y, count_left, count_right);
                
                if (count_left % 2 == 0 || count_right % 2 == 0) continue;
                else 
                {
                    W_intersect_array[threadId] = 1;
                    return;
                }
            } 
            
            if (W_intersect_array[threadId] == 0) Need_intersect_array[threadId] = 1; 
        }
    }
    """
    # INPUT 1:
    W_array = np.asarray(W_list, np.int32)
    W_array_GPU = gpuarray.to_gpu(W_array)
    
    # --------------------------------- Preparing for the INPUT 2-11 --------------------------------- 
    # mbr of selected polygon pairs
    shp_A_mbr = np.zeros((len(W_list),4))
    shp_B_mbr = np.zeros((len(W_list),4))
    
    # vertices of selected shape A and B
    shp_A_points = list()
    shp_B_points = list()
    
    # index of vertices of selected shape A and B
    shp_A_points_index = list()
    shp_B_points_index = list()
    
    # edges of selected shape A and B
    shp_A_edges = list()
    shp_B_edges = list()
    
    # index of edges of selected shape A and B
    shp_A_edges_index = list()
    shp_B_edges_index = list()
    
    for index, (plyn_A_id, plyn_B_id) in enumerate(W_list):
        # mbr of selected polygon pairs
        shp_A_mbr[index][:] = shp_A_array[plyn_A_id]
        shp_B_mbr[index][:] = shp_B_array[plyn_B_id]
        
        # vertices of selected shape A and B
        shp_A_points.append(shp_A[plyn_A_id].points[:-1])
        shp_B_points.append(shp_B[plyn_B_id].points[:-1])
        shp_A_points_index.append(len(shp_A[plyn_A_id].points)-1) # the starting point and ending point are the same, so double-counted
        shp_B_points_index.append(len(shp_B[plyn_B_id].points)-1) # the starting point and ending point are the same, so double-counted
        
        # edges of selected shape A and B
        shp_A_edges.append(points2edges(shp_A[plyn_A_id].points))
        shp_A_edges_index.append(len(shp_A[plyn_A_id].points)-1) # The number of edge is 1 less than points
        shp_B_edges.append(points2edges(shp_B[plyn_B_id].points))
        shp_B_edges_index.append(len(shp_B[plyn_B_id].points)-1) # The number of edge is 1 less than points
    
    # flatten list
    shp_A_mbr = list(itertools.chain.from_iterable(shp_A_mbr))
    shp_B_mbr = list(itertools.chain.from_iterable(shp_B_mbr))
    
    shp_A_points = list(itertools.chain.from_iterable(shp_A_points))
    shp_B_points = list(itertools.chain.from_iterable(shp_B_points))
    shp_A_edges = list(itertools.chain.from_iterable(shp_A_edges))
    shp_B_edges = list(itertools.chain.from_iterable(shp_B_edges))
        
    shp_A_points_index = np.add.accumulate(np.asarray(shp_A_points_index))
    shp_B_points_index = np.add.accumulate(np.asarray(shp_B_points_index))
    shp_A_edges_index = np.add.accumulate(np.asarray(shp_A_edges_index))
    shp_B_edges_index = np.add.accumulate(np.asarray(shp_B_edges_index))
    # ----------------------------------------------------------------------------------------------- 

    # INPUT 2
    shp_A_mbr = np.asarray(shp_A_mbr, np.float32)
    shp_A_mbr_GPU = gpuarray.to_gpu(shp_A_mbr)

    # INPUT 3
    shp_B_mbr = np.asarray(shp_B_mbr, np.float32)
    shp_B_mbr_GPU = gpuarray.to_gpu(shp_B_mbr)

    # INPUT 4
    shp_A_points = np.asarray(shp_A_points, np.float32)
    shp_A_points_GPU = gpuarray.to_gpu(shp_A_points)
    
    # INPUT 5
    shp_B_points = np.asarray(shp_B_points, np.float32)
    shp_B_points_GPU = gpuarray.to_gpu(shp_B_points)
    
    # INPUT 6
    shp_A_points_index = np.asarray(shp_A_points_index, np.int32)
    shp_A_points_index_GPU = gpuarray.to_gpu(shp_A_points_index)
    
    # INPUT 7
    shp_B_points_index = np.asarray(shp_B_points_index, np.int32)
    shp_B_points_index_GPU = gpuarray.to_gpu(shp_B_points_index)
    
    # INPUT 8
    shp_A_edges = np.asarray(shp_A_edges, np.float32)
    shp_A_edges_GPU = gpuarray.to_gpu(shp_A_edges)
    
    # INPUT 9
    shp_B_edges = np.asarray(shp_B_edges, np.float32)
    shp_B_edges_GPU = gpuarray.to_gpu(shp_B_edges)

    # INPUT 10
    shp_A_edges_index = np.asarray(shp_A_edges_index, np.int32)
    shp_A_edges_index_GPU = gpuarray.to_gpu(shp_A_edges_index)
    
    # INPUT 11
    shp_B_edges_index = np.asarray(shp_B_edges_index, np.int32)
    shp_B_edges_index_GPU = gpuarray.to_gpu(shp_B_edges_index)   
    
    # OUTPUT 1
    W_intersect_array = np.zeros(len(W_list),dtype=np.int32)
    W_intersect_array_GPU = gpuarray.to_gpu(W_intersect_array)

    # OUTPUT 2
    Need_intersect_array = np.zeros(len(W_list),dtype=np.int32)
    Need_intersect_array_GPU = gpuarray.to_gpu(Need_intersect_array)
    
    # Start using GPU
    max_block_size = 512
    if len(W_list)<=max_block_size:
        max_block_size = len(W_list)
        max_grid_size = 1
    else:
        max_grid_size = (len(W_list)/max_block_size)+1
    
    # get the kernel function
    mod = compiler.SourceModule(kernel_code)
    func_count = mod.get_function("PnPtest")
    
    func_count(
         # input
         W_array_GPU, 
         shp_A_mbr_GPU, shp_B_mbr_GPU,
         shp_A_points_GPU, shp_B_points_GPU,
         shp_A_points_index_GPU, shp_B_points_index_GPU,
         shp_A_edges_GPU, shp_B_edges_GPU,
         shp_A_edges_index_GPU, shp_B_edges_index_GPU,
         # output
         W_intersect_array_GPU, Need_intersect_array_GPU,
         # some fixed variables
         np.int32(len(W_list)), 
         # block size
         block=(max_block_size,1,1), 
         # gird size
         grid=(max_grid_size,1,1))
    
    # get W_intersect array and need more edge intersect test array
    W_intersect_array = W_intersect_array_GPU.get()
    Need_intersect_array = Need_intersect_array_GPU.get()
    
    W_intersect_result = [W_list[i] for i,intersect_value in enumerate(W_intersect_array) if intersect_value == 1]
    Need_intersect_test = [W_list[i] for i,intersect_value in enumerate(Need_intersect_array) if intersect_value == 1]
    
    return W_intersect_result,Need_intersect_test


def edge_dic2edge_array(plyn_poten_edges):
    """ convert edge dictionary from CMF to GPU readable array
    
    Parameters
    ----------
    plyn_poten_edges: {'41-1':[2,3,4]} means in the 41st pair from filter, the 1 polygon,
                                       has the 2nd,3rd,4th edge intersect with MBR, 
                                       that should be consider in the edge intersect test
    
    Returns
    ----------  
    plyn_pairid_list: the id for the pair in from filter result
    plyn_id_list: which polygon
    edge_id_list: which edges should do the intersection test
    edge_accu_list: for GPU find which range for the current thread should take a look
    """
    plyn_pairid_list = list()
    plyn_id_list = list()
    edge_id_list = list()
    edge_accu_list = list()
    
    for key in sorted(plyn_poten_edges):
        
        pair_id = int(key.split("-")[0])
        plyn_id = int(key.split("-")[1])
        
        plyn_pairid_list.append(pair_id)
        plyn_id_list.append(plyn_id)
        edge_id_list.append(plyn_poten_edges[key])
        edge_accu_list.append(len(plyn_poten_edges[key]))

    return np.asarray(plyn_pairid_list), \
           np.asarray(plyn_id_list), \
           np.asarray([y for x in edge_id_list for y in x]), \
           np.add.accumulate(np.asarray(edge_accu_list))


def GPU_W_list_EI_test(shp_A,shp_B,W_list):
    
    """ Edge-intersection test for I list candidate
    
    Parameters
    ----------
    shp_A: the read shapefile of A with information about each vertice
    shp_B: the read shapefile of B with information about each vertice
    Need_intersect_test: leftover pairs after W list went through PntInPolygon test
    
    Returns
    ----------  
    intersect_pair_list: for pairs in Need_intersect_test list that are intersect
    """
    # KERNEL FUNCTION
    # Please Note: In C programming: Maths operation only happen in same data type
    kernel_code = """    // line 0
    #include <stdio.h>   // line 1
    #include <math.h>    // line 2 ...
     
    __global__ void EI_test(int *W_array, \
                            float *shp_A_edges, float *shp_B_edges, \
                            int *shp_A_edges_index, int *shp_B_edges_index, \
                            int *W_intersect_array, \
                            int len_W_list)
    {
        int threadId = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
        
        int plyn_A_edge_start,plyn_A_edge_end;
        int plyn_B_edge_start,plyn_B_edge_end;
        int plyn_A_edge_index,plyn_B_edge_index;
        int intersect_index;
        
        float A_X, A_Y, B_X, B_Y; // easier for line intersection
        float C_X, C_Y, D_X, D_Y; // easier for line intersection
        int ccw_ACD, ccw_BCD, ccw_ABC, ccw_ABD; // for line intersection
        
        // stop extra threads
        if (threadId>=len_W_list) return;
        
        // find the index range
        if (threadId==0) 
        {
            plyn_A_edge_start = 0;
            plyn_A_edge_end = shp_A_edges_index[0];
            plyn_B_edge_start = 0;
            plyn_B_edge_end = shp_B_edges_index[0];
        }
        
        else
        {
            plyn_A_edge_start = shp_A_edges_index[threadId-1];
            plyn_A_edge_end = shp_A_edges_index[threadId];
            plyn_B_edge_start = shp_B_edges_index[threadId-1];
            plyn_B_edge_end = shp_B_edges_index[threadId];
        }
        
        /* Edge intersect Test*/
        // go through each edge from shp A
        for (plyn_A_edge_index = plyn_A_edge_start; plyn_A_edge_index<plyn_A_edge_end; plyn_A_edge_index++)
        {
            A_X = shp_A_edges[plyn_A_edge_index*4];
            A_Y = shp_A_edges[plyn_A_edge_index*4+1];
            B_X = shp_A_edges[plyn_A_edge_index*4+2];
            B_Y = shp_A_edges[plyn_A_edge_index*4+3];
            
            // go through each edge from shp B
            for (plyn_B_edge_index = plyn_B_edge_start; plyn_B_edge_index<plyn_B_edge_end; plyn_B_edge_index++)
            {
                C_X = shp_B_edges[plyn_B_edge_index*4];
                C_Y = shp_B_edges[plyn_B_edge_index*4+1];
                D_X = shp_B_edges[plyn_B_edge_index*4+2];
                D_Y = shp_B_edges[plyn_B_edge_index*4+3];
                
                // implement the ccw and intersect function; eg: total = (total<2500)?(1):(10);
                ccw_ACD = ( ( (D_Y-A_Y)*(C_X-A_X) ) > ( (C_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                ccw_BCD = ( ( (D_Y-B_Y)*(C_X-B_X) ) > ( (C_Y-B_Y)*(D_X-B_X) ) )?(1):(0);
                ccw_ABC = ( ( (C_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(C_X-A_X) ) )?(1):(0);
                ccw_ABD = ( ( (D_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                
                // check if they are intersected
                intersect_index = ((ccw_ACD!=ccw_BCD)&&(ccw_ABC!=ccw_ABD))?(1):(0);
                
                // write output
                if (intersect_index) 
                {
                    W_intersect_array[threadId]=1;
                    return;
                }
            }
        }
    }
    """
    # INPUT 1:
    W_array = np.asarray(W_list, np.int32)
    W_array_GPU = gpuarray.to_gpu(W_array)
    
    # --------------------------------- Preparing for the INPUT 2-11 ---------------------------------
    # edges of selected shape A and B
    shp_A_edges = list()
    shp_B_edges = list()
    
    # index of edges of selected shape A and B
    shp_A_edges_index = list()
    shp_B_edges_index = list()
    
    for plyn_A_id, plyn_B_id in W_list:
        
        # read edge candidates from I list 
        plyn_A_all_edges = points2edges(shp_A[plyn_A_id].points)
        plyn_B_all_edges = points2edges(shp_B[plyn_B_id].points)
        
        # collect edges to one list
        shp_A_edges.append(plyn_A_all_edges)
        shp_B_edges.append(plyn_B_all_edges)
        shp_A_edges_index.append(len(plyn_A_all_edges))
        shp_B_edges_index.append(len(plyn_B_all_edges))
    
    # flatten list
    shp_A_edges = list(itertools.chain.from_iterable(shp_A_edges))
    shp_B_edges = list(itertools.chain.from_iterable(shp_B_edges))        
    shp_A_edges_index = np.add.accumulate(np.asarray(shp_A_edges_index))
    shp_B_edges_index = np.add.accumulate(np.asarray(shp_B_edges_index))   
    # -----------------------------------------------------------------------------------------------
    
    # INPUT 2
    shp_A_edges = np.asarray(shp_A_edges, np.float32)
    shp_A_edges_GPU = gpuarray.to_gpu(shp_A_edges)
    
    # INPUT 3
    shp_B_edges = np.asarray(shp_B_edges, np.float32)
    shp_B_edges_GPU = gpuarray.to_gpu(shp_B_edges)

    # INPUT 4
    shp_A_edges_index = np.asarray(shp_A_edges_index, np.int32)
    shp_A_edges_index_GPU = gpuarray.to_gpu(shp_A_edges_index)
    
    # INPUT 5
    shp_B_edges_index = np.asarray(shp_B_edges_index, np.int32)
    shp_B_edges_index_GPU = gpuarray.to_gpu(shp_B_edges_index)   
    
    # OUTPUT 1
    W_intersect_array = np.zeros(len(W_list),dtype=np.int32)
    W_intersect_array_GPU = gpuarray.to_gpu(W_intersect_array) 
    
    # Start using GPU
    max_block_size = 512
    if len(W_list)<=max_block_size:
        max_block_size = len(W_list)
        max_grid_size = 1
    else:
        max_grid_size = (len(W_list)/max_block_size)+1
    
    # get the kernel function
    mod = compiler.SourceModule(kernel_code)
    func_count = mod.get_function("EI_test")
    
    func_count(
         # input
         W_array_GPU, 
         shp_A_edges_GPU, shp_B_edges_GPU,
         shp_A_edges_index_GPU, shp_B_edges_index_GPU,
         # output
         W_intersect_array_GPU,
         # some fixed variables
         np.int32(len(W_list)), 
         # block size
         block=(max_block_size,1,1), 
         # gird size
         grid=(max_grid_size,1,1))
    
    # get W_intersect array and need more edge intersect test array
    W_intersect_array = W_intersect_array_GPU.get()
    W_intersect_result = [W_list[i] for i,intersect_value in enumerate(W_intersect_array) if intersect_value == 1]
    
    return W_intersect_result


def I_list_EI_test(shp_A,shp_B,
                   I_list,plyn_A_poten_edges,plyn_B_poten_edges):
    """ Edge-intersection test for I list candidate
    
    Focus on dealing with I_list with candidate pair edges
    
    Parameters
    ----------
    shp_A: the read shapefile of A with information about each vertice
    shp_B: the read shapefile of B with information about each vertice
    I_list: for pairs in I list, they will be tested with simple intersection 
            and save potential edges for each candidate polygon for final "edge intersection"
    plyn_A_edges: tells which edge from polygon from shapefile A needs to be checked
                  in this way, lots of edges do not need to check at all
    plyn_B_edges: tells which edge from polygon from shapefile A needs to be checked
                  in this way, lots of edges do not need to check at all
    REFERENCE: plyn_A_poten_edges_dict["%d-%d"%(pair_id,plyn_A_id)] = plyn_A_edges_index
               plyn_B_poten_edges_dict["%d-%d"%(pair_id,plyn_B_id)] = plyn_B_edges_index 
    
    Returns
    ----------  
    intersect_pair_list: for pairs in I_list that are intersect
    """
    # the list to store output result
    I_intersect_result = list()
    
    for index, (plyn_A_id, plyn_B_id) in enumerate(I_list):
        
        plyn_A_all_edges = points2edges(shp_A[plyn_A_id].points)
        plyn_B_all_edges = points2edges(shp_B[plyn_B_id].points)
        
        plyn_A_filtered_edges_index = plyn_A_poten_edges['%d-%d'%(index,plyn_A_id)]
        plyn_B_filtered_edges_index = plyn_B_poten_edges['%d-%d'%(index,plyn_B_id)]
        
        plyn_A_filtered_edges = [plyn_A_all_edges[i] for i in plyn_A_filtered_edges_index]
        plyn_B_filtered_edges = [plyn_B_all_edges[i] for i in plyn_B_filtered_edges_index]
        
        intersect_index = 0
        
        for plyn_A_start_x, plyn_A_start_y, plyn_A_end_x, plyn_A_end_y in plyn_A_filtered_edges:
            for plyn_B_start_x, plyn_B_start_y, plyn_B_end_x, plyn_B_end_y in plyn_B_filtered_edges:
                    
                a = Point(plyn_A_start_x,plyn_A_start_y)
                b = Point(plyn_A_end_x,plyn_A_end_y)
                c = Point(plyn_B_start_x,plyn_B_start_y)
                d = Point(plyn_B_end_x,plyn_B_end_y)
                
                if intersect(a,b,c,d) and [plyn_A_id,plyn_B_id] not in I_intersect_result:
                    intersect_index = 1
                    I_intersect_result.append([plyn_A_id,plyn_B_id])
                    break
            
            if intersect_index:
                break
    
    return I_intersect_result
 

def GPU_I_list_EI_test(shp_A,shp_B,
                       I_list,
                       plyn_A_poten_edges,plyn_B_poten_edges):
    
    """ GPU accelerated Edge-intersection test for I list candidate
    
    Focus on dealing with I_list with candidate pair edges
    
    Parameters
    ----------
    shp_A: the read shapefile of A with information about each vertice
    shp_B: the read shapefile of B with information about each vertice
    I_list: for pairs in I list, they will be tested with simple intersection 
            and save potential edges for each candidate polygon for final "edge intersection"
    plyn_A_edges: tells which edge from polygon from shapefile A needs to be checked
                  in this way, lots of edges do not need to check at all
    plyn_B_edges: tells which edge from polygon from shapefile A needs to be checked
                  in this way, lots of edges do not need to check at all
    REFERENCE: plyn_A_poten_edges_dict["%d-%d"%(pair_id,plyn_A_id)] = plyn_A_edges_index
               plyn_B_poten_edges_dict["%d-%d"%(pair_id,plyn_B_id)] = plyn_B_edges_index 
    
    Returns
    ----------  
    intersect_pair_list: for pairs in I_list that are intersect
    """
    # KERNEL FUNCTION
    # Please Note: In C programming: Maths operation only happen in same data type
    kernel_code = """    // line 0
    #include <stdio.h>   // line 1
    #include <math.h>    // line 2 ...
     
    __global__ void EI_test(int *I_array, \
                            float *shp_A_edges, float *shp_B_edges, \
                            int *shp_A_edges_index, int *shp_B_edges_index, \
                            int *I_intersect_array, \
                            int len_I_list)
    {
        int threadId = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));
        
        int plyn_A_edge_start,plyn_A_edge_end;
        int plyn_B_edge_start,plyn_B_edge_end;
        int plyn_A_edge_index,plyn_B_edge_index;
        int intersect_index;
        
        float A_X, A_Y, B_X, B_Y; // easier for line intersection
        float C_X, C_Y, D_X, D_Y; // easier for line intersection
        int ccw_ACD, ccw_BCD, ccw_ABC, ccw_ABD; // for line intersection
        
        // stop extra threads
        if (threadId>=len_I_list) return;
        
        // find the index range
        if (threadId==0) 
        {
            plyn_A_edge_start = 0;
            plyn_A_edge_end = shp_A_edges_index[0];
            plyn_B_edge_start = 0;
            plyn_B_edge_end = shp_B_edges_index[0];
        }
        
        else
        {
            plyn_A_edge_start = shp_A_edges_index[threadId-1];
            plyn_A_edge_end = shp_A_edges_index[threadId];
            plyn_B_edge_start = shp_B_edges_index[threadId-1];
            plyn_B_edge_end = shp_B_edges_index[threadId];
        }
        
        /* Edge intersect Test*/
        // go through each edge from shp A
        for (plyn_A_edge_index = plyn_A_edge_start; plyn_A_edge_index<plyn_A_edge_end; plyn_A_edge_index++)
        {
            A_X = shp_A_edges[plyn_A_edge_index*4];
            A_Y = shp_A_edges[plyn_A_edge_index*4+1];
            B_X = shp_A_edges[plyn_A_edge_index*4+2];
            B_Y = shp_A_edges[plyn_A_edge_index*4+3];
            
            // go through each edge from shp B
            for (plyn_B_edge_index = plyn_B_edge_start; plyn_B_edge_index<plyn_B_edge_end; plyn_B_edge_index++)
            {
                C_X = shp_B_edges[plyn_B_edge_index*4];
                C_Y = shp_B_edges[plyn_B_edge_index*4+1];
                D_X = shp_B_edges[plyn_B_edge_index*4+2];
                D_Y = shp_B_edges[plyn_B_edge_index*4+3];
                
                // implement the ccw and intersect function; eg: total = (total<2500)?(1):(10);
                ccw_ACD = ( ( (D_Y-A_Y)*(C_X-A_X) ) > ( (C_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                ccw_BCD = ( ( (D_Y-B_Y)*(C_X-B_X) ) > ( (C_Y-B_Y)*(D_X-B_X) ) )?(1):(0);
                ccw_ABC = ( ( (C_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(C_X-A_X) ) )?(1):(0);
                ccw_ABD = ( ( (D_Y-A_Y)*(B_X-A_X) ) > ( (B_Y-A_Y)*(D_X-A_X) ) )?(1):(0);
                
                // check if they are intersected
                intersect_index = ((ccw_ACD!=ccw_BCD)&&(ccw_ABC!=ccw_ABD))?(1):(0);
                
                // write output
                if (intersect_index) 
                {
                    I_intersect_array[threadId]=1;
                    return;
                }
            }
        }
    }
    """
    # INPUT 1:
    I_array = np.asarray(I_list, np.int32)
    I_array_GPU = gpuarray.to_gpu(I_array)
    
    # --------------------------------- Preparing for the INPUT 2-11 ---------------------------------
    # edges of selected shape A and B
    shp_A_edges = list()
    shp_B_edges = list()
    
    # index of edges of selected shape A and B
    shp_A_edges_index = list()
    shp_B_edges_index = list()
    
    for index, (plyn_A_id, plyn_B_id) in enumerate(I_list):
        
        # read edge candidates from I list 
        plyn_A_all_edges = points2edges(shp_A[plyn_A_id].points)
        plyn_B_all_edges = points2edges(shp_B[plyn_B_id].points)
        plyn_A_filtered_edges_index = plyn_A_poten_edges['%d-%d'%(index,plyn_A_id)]
        plyn_B_filtered_edges_index = plyn_B_poten_edges['%d-%d'%(index,plyn_B_id)]
        plyn_A_filtered_edges = [plyn_A_all_edges[i] for i in plyn_A_filtered_edges_index]
        plyn_B_filtered_edges = [plyn_B_all_edges[i] for i in plyn_B_filtered_edges_index]
        
        # collect edges to one list
        shp_A_edges.append(plyn_A_filtered_edges)
        shp_B_edges.append(plyn_B_filtered_edges)
        shp_A_edges_index.append(len(plyn_A_filtered_edges))
        shp_B_edges_index.append(len(plyn_B_filtered_edges))
    
    # flatten list
    shp_A_edges = list(itertools.chain.from_iterable(shp_A_edges))
    shp_B_edges = list(itertools.chain.from_iterable(shp_B_edges))        
    shp_A_edges_index = np.add.accumulate(np.asarray(shp_A_edges_index))
    shp_B_edges_index = np.add.accumulate(np.asarray(shp_B_edges_index))   
    # -----------------------------------------------------------------------------------------------
    
    # INPUT 2
    shp_A_edges = np.asarray(shp_A_edges, np.float32)
    shp_A_edges_GPU = gpuarray.to_gpu(shp_A_edges)
    
    # INPUT 3
    shp_B_edges = np.asarray(shp_B_edges, np.float32)
    shp_B_edges_GPU = gpuarray.to_gpu(shp_B_edges)

    # INPUT 4
    shp_A_edges_index = np.asarray(shp_A_edges_index, np.int32)
    shp_A_edges_index_GPU = gpuarray.to_gpu(shp_A_edges_index)
    
    # INPUT 5
    shp_B_edges_index = np.asarray(shp_B_edges_index, np.int32)
    shp_B_edges_index_GPU = gpuarray.to_gpu(shp_B_edges_index)   
    
    # OUTPUT 1
    I_intersect_array = np.zeros(len(I_list),dtype=np.int32)
    I_intersect_array_GPU = gpuarray.to_gpu(I_intersect_array) 
    
    # Start using GPU
    max_block_size = 512
    if len(I_list)<=max_block_size:
        max_block_size = len(I_list)
        max_grid_size = 1
    else:
        max_grid_size = (len(I_list)/max_block_size)+1
    
    # get the kernel function
    mod = compiler.SourceModule(kernel_code)
    func_count = mod.get_function("EI_test")
    
    func_count(
         # input
         I_array_GPU, 
         shp_A_edges_GPU, shp_B_edges_GPU,
         shp_A_edges_index_GPU, shp_B_edges_index_GPU,
         # output
         I_intersect_array_GPU,
         # some fixed variables
         np.int32(len(I_list)), 
         # block size
         block=(max_block_size,1,1), 
         # gird size
         grid=(max_grid_size,1,1))
    
    # get W_intersect array and need more edge intersect test array
    I_intersect_array = I_intersect_array_GPU.get()
    I_intersect_result = [I_list[i] for i,intersect_value in enumerate(I_intersect_array) if intersect_value == 1]
    
    return I_intersect_result