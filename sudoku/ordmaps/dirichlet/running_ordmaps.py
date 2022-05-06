from ordmaps.dirchlet.discrete_gradient_ascent import gradient_ascent

from minimise_angle.davidnewordmap import davidnew

def ordmap(matrix,method,x0='Daniel',depth=1,seeded=True,generating_set='neighbourtranspositions'):
    if method=='David_gradient':
        matrix=davidnew(matrix)
    if method=='gradient' or method=='David_gradient':
        if seeded==True:
            return gradient_ascent_seeded(matrix,x0=x0,depth=depth, generating_set=generating_set)
        if seeded==False:
            return gradient_ascent(matrix,x0=x0,depth=depth, generating_set=generating_set)
    if method=='David':
        return davidnew(matrix)