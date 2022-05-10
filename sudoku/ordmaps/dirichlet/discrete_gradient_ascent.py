import numpy as np
import sys
sys.path.append('../')
from sudoku.ordmaps.dirichlet.generating_sets import *
from sudoku.ordmaps.matrix_permutation_auxiliaryfunctions import *

from functools import cmp_to_key #For user defined sorting
#from minimise_angle.davidnewordmap import davidnew
# last import must be adapted after update of minimise_angle.davidnewordmap...

#https://stackoverflow.com/questions/5213033/sort-a-list-of-lists-with-a-custom-compare-function
###Generating useful classes of matrices in the permutation group###

#This function generates entries for a matrix representating the permutation (k,m)

##
#This function returns 1 if y<x, -1 y>x and 0 for equality in a metric determined by x0
###Gradient Ascent:###

def gradient_ascent(x,x0,gen_name='neighbourtranspositions'):
    matrix_dim=np.shape(x)


    gen_row=generators(gen_name,matrix_dim,'row').elements
    gen_col=generators(gen_name,matrix_dim,'col').elements
    trans=generators(gen_name,matrix_dim,'trans').elements


    len_row=len(gen_row)
    len_col=len(gen_col)
    len_trans=len(trans)

    max_found=False
    while max_found==False:
        i=0
        while i<len_row+len_col+len_trans:
            if i<len_row:
                y = np.dot(gen_row[i], x)
            if i>=len_row and i<len_row+len_col:
                y = np.dot(x, gen_col[i-len_row])
            if i>len_row+len_col:
                y=trans[i-len_row-len_col](x)

            if matrix_order(x, y, x0) == -1:  # In this case y>x
                x = y
                i = 0
            else:
                i = i + 1
        max_found=True
    return x

# We produce seed matrices by applying cyclic permutations to the start matrix and check which one achieves the highest value.

def gradient_ascent_seeded(x,x0, gen_name='neighbourtranspositions'):
    k, m = np.shape(x)
    seeded_ascents=np.array([[gradient_ascent(np.dot(np.dot(cycle(i,k),x),cycle(j,m)),x0, gen_name=gen_name) for i in range(k)]  for j in range(m)])
    if x0=='Daniel':
    #Here we are faced with the task to find the maximum of a list for a non-standard ordering. cmp_to_key allows for
    #self defined orderings
        seeds_sorted=sorted(flatten_onelevel(seeded_ascents), key=cmp_to_key(lambda x,y: matrix_order(x,y,x0)))
        return seeds_sorted[-1]
    else:
        array_of_norms=np.array([[matrix_innerproduct(seeded_ascents[j][i],x0) for i in range(k)] for j in range(m)])
        max_seed=argmax_nonflat(array_of_norms)
        return seeded_ascents[max_seed]

#We permute our original matrix n_tests times and see how often we end up at the same local max.
# This is an indicator for when we have found a global max, if n_test suff high.

if __name__ == '__main__':
    #print(gradient_ascent_seeded(np.array([[0,1],[2,0]]),x0='Daniel'))
    A=np.array([[0,0,3,0,2,0,6,0,0],[9,0,0,3,0,5,0,0,1],[0,0,1,8,0,6,4,0,0],
                [0,0,8,1,0,2,9,0,0],[7,0,0,0,0,0,0,0,8],[0,0,6,7,0,8,2,0,0],
                [0,0,2,6,0,9,5,0,0],[8,0,0,2,0,3,0,0,9],[0,0,5,0,1,0,3,0,0]])
    print(A)
    print(gradient_ascent_seeded(A, x0='Daniel',gen_name='sudoku'))



