"""
Author:    Weixing Zhang
Date:      2018
Name:      Spatial join filter
"""

# modules
import os.path
import timeit
import functions as fn

# file path 
root = r'C:\Users\downloads'
polygon_A_path = os.path.join(root, '2017_CT_Mansfield_TestPolygon_1.shp')
polygon_B_path = os.path.join(root, '2017_CT_Mansfield_TestPolygon_2.shp')

# __________________read data__________________ 
shp_A, shp_A_array = fn.read_shp(polygon_A_path)
shp_B, shp_B_array = fn.read_shp(polygon_B_path)

# __________________Sort MBR Filter__________________ 
filter_index_xy = fn.GPU_serial_sorted_MBR_filter(shp_A_array,shp_B_array)

# __________________Common MBR Filter__________________
W_list, I_list, plyn_A_poten_edges, plyn_B_poten_edges = fn.GPU_common_MBR_filter(shp_A_array,shp_B_array,
                                                                                  shp_A,shp_B,filter_index_xy)

# __________________W LIST_______________________
# (1) Refinement for W_list
W_intersect_result, Need_intersect_test = fn.GPU_PnPtest(shp_A_array,shp_B_array,
                                                         shp_A,shp_B,W_list)
# (2) W edges leftover for intersect test
W_Need_intersect_reult = fn.GPU_W_list_EI_test(shp_A,shp_B,Need_intersect_test)

# __________________I LIST_______________________ 
I_intersect_reult = fn.GPU_I_list_EI_test(shp_A,shp_B,I_list,plyn_A_poten_edges,plyn_B_poten_edges)

# __________________RESULT__________________
print "FINAL RESULT:"
print W_intersect_result+W_Need_intersect_reult+I_intersect_reult


