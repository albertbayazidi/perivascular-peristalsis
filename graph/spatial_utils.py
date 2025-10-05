import numpy as np


def compute_vessel_endpoint (previousvessel, surfacenormal,angle,length) :
    """ From a previous vessel defined in 3D, the brain surface at the end of the previous vessel, angle and length : 
        compute the coordinate of the end node of the current vessel """

    # project the previous vessel in the current plane
    # and get the direction vector of the previous vessel 
    pm1=previousvessel[0]
    p0=previousvessel[1]

    vector_previous=[p0[i]-pm1[i] for i in range(len(pm1))]

    previousdir = project_onto_plane(vector_previous, surfacenormal)


    # compute the direction vector of the new vessel with the angle
    newdir=rotate_in_plane(previousdir,surfacenormal,angle)

    # compute the location of the end of the new vessel
    pnew=translation(p0,newdir,length)


    return pnew


def normalize(x):
    return [x[i] / np.linalg.norm(x) for i in range(len(x))]

def rotate_in_plane(x,n,angle):
    """ angle : rotation degree """

    rotation_radians = np.radians(angle)

    rotation_axis = np.array(n)

    rotation_vector = rotation_radians * rotation_axis

    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_rotvec(rotation_vector)

    rotated_vec = rotation.apply(x)

    return rotated_vec

def project_onto_plane(x, n):
    d = np.dot(x, n) / np.linalg.norm(n)
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]

def translation(p0,direction,length) :
    #normalise the direction vector
    direction=normalize(direction)

    # compute the location of the end of the new vessel
    pnew=[(p0[i]+direction[i]*length) for i in range(len(p0))]
    return pnew

def orientation(p, q, r):
    '''
    to find the orientation of an ordered triplet (p,q,r)
    function returns the following values:
    0 : Collinear points
    1 : Clockwise points
    2 : Counterclockwise

    See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    for details of below formula.
    '''

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:

        # Clockwise orientation
        return 1
    elif val < 0:

        # Counterclockwise orientation
        return 2
    else:

        # Collinear orientation
        return 0


# This code is contributed by Ansh Riyal
def doIntersect(p1, q1, p2, q2):

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2, q1, q2):
        return True

    # If none of the cases
    return False

def onSegment(p, q, r):
    if (
        (q.x <= max(p.x, r.x))
        and (q.x >= min(p.x, r.x))
        and (q.y <= max(p.y, r.y))
        and (q.y >= min(p.y, r.y))
    ):
        return True
    return False

class Point:
    def __init__(self, xx):
        self.x = xx[0]
        self.y = xx[1]
