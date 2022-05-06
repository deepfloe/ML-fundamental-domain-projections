def ordmap_invariance(x,method,x0, n_tests=10,seeded=True,generating_set='neighbourtranspositions'):
    def ordmap_local(matrix):
        return ordmap(matrix,method=method, x0=x0, seeded=seeded, generating_set=generating_set)
    y = ordmap_local(x)
    list_equal = [matrix_equality(ordmap_local(permuteMatrix(x)), y) for i in range(n_tests)]
    n_equal = np.sum(list_equal)
    return n_equal

def gradient_ascent_invariance(x,x0,n_tests=10,seeded=True,generating_set='neighbourtranspositions'):
    return ordmap_invariance(x,'gradient',x0=x0,n_tests=n_tests,seeded=seeded,generating_set=generating_set)

##The following function prints out matrices where the gradient ascent gets stuck in a local max
def print_troublemakers(matrixlist,x0,n_tests=10,seeded=True):
    i=0
    j=1
    while i<len(matrixlist) and j<6:
        n_equal=gradient_ascent_invariance(matrixlist[i],x0,n_tests,seeded)
        if n_equal<n_tests:
            print(n_equal)
            print(matrixlist[i])
            j=j+1
        i=i+1


##This is to check how many matrices achieve the maximum of gradient_ascent_invariance
def ordmap_invariance_list(matrixlist,method,x0,n_tests=10,seeded=True,generating_set='neighbourtranspositions'):
    testlist=[ordmap_invariance(x,method,x0,n_tests=n_tests,seeded=seeded,generating_set=generating_set) for x in matrixlist]
    return testlist.count(n_tests)

