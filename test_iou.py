import numpy as np
from scipy.spatial import ConvexHull

def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return(outputList)
    
def corners_from_box(box, param6=True, with_translation=False):
    # box given as: [w, l, h, cx, cy, cz, z]
    if param6:
        w, l, h, cx, cy, cz = box
    else:
        w, l, h, cx, cy, cz, _ = box

    (tx, ty, tz) = (cx, cy, cz) if with_translation else (0,0,0)

    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.dot(np.eye(3), np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + ty
    corners_3d[1,:] = corners_3d[1,:] + tz
    corners_3d[2,:] = corners_3d[2,:] + tx
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0
    
def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def box3d_iou(box1, box2, param6=True, with_translation=False):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    corners1 = corners_from_box(box1, param6, with_translation)
    corners2 = corners_from_box(box2, param6, with_translation)

    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)]

    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)

    volmin = min(vol1, vol2)

    iou = inter_vol / volmin #(vol1 + vol2 - inter_vol)

    return iou, iou_2d

def corners_from_box_up_z_axis(box):
    corner_offsets = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
            
    l,w,h,c_x,c_y,c_z = box
    half_l, half_w, half_h = l / 2, w / 2, h / 2

    corners = []
    for offset in corner_offsets:
        corner_x = c_x + offset[0] * half_l
        corner_y = c_y + offset[1] * half_w
        corner_z = c_z + offset[2] * half_h
        corners.append((corner_x, corner_y, corner_z))
    corners = np.array(corners)

    bbox_min = corners[0]
    bbox_max = corners[6]

    corners2 = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
    ])
    print(corners)
    print(corners2)
    print(corners == corners)
    return corners, corners, max

def corners_from_box_up_z_axis_simple(box):
    corner_offsets = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
            
    l,w,h,c_x,c_y,c_z = box
    half_l, half_w, half_h = l / 2, w / 2, h / 2

    corners = []
    for offset in corner_offsets:
        corner_x = c_x + offset[0] * half_l
        corner_y = c_y + offset[1] * half_w
        corner_z = c_z + offset[2] * half_h
        corners.append((corner_x, corner_y, corner_z))
    corners = np.array(corners)

    bbox_min = corners[0]
    bbox_max = corners[6]

    return corners

def check_if_has_rel(box1, box2, type):
    #input of box: l, w, h, x, y, z
    corners1 = corners_from_box_up_z_axis_simple(box1)
    corners2 = corners_from_box_up_z_axis_simple(box2)

    if type=="left": #check if box1 - left - box2
        box1_center_x = box1[3]
        box2_min_x = corners2[0][0]
        return True if box1_center_x < box2_min_x else False
    elif type=="right":
        box1_center_x = box1[3]
        box2_max_x = corners2[6][0]
        return True if box1_center_x > box2_max_x else False
    elif type=="front":
        box1_center_y = box1[4]
        box2_min_y = corners2[0][1]
        return True if box1_center_y < box2_min_y else False
    elif type=="behind":
        box1_center_y = box1[4]
        box2_max_y = corners2[6][1]
        return True if box1_center_y > box2_max_y else False


if __name__ == "__main__":
    box1 = [                0.12468190491199493,
        0.08774815499782562,
        0.1450921595096588,
        -0.14064285159111023,
        -0.14306005835533142,
        0.3946387767791748]  
    box2 = [
        0.1481591910123825,
        0.15000000596046448,
        0.08076509833335876,
        -0.18441036343574524,
        0.036920733749866486,
        0.3624652624130249
    ]
    print(check_if_has_rel(box1,box2,"front"))
    print(check_if_has_rel(box1,box2,"behind"))
    print(check_if_has_rel(box1,box2,"left"))
    print(check_if_has_rel(box1,box2,"right"))
    #corners_from_box_up_z_axis(box1)
    #box2 = [ 0.699999988079071, 0.6299999952316284, 0.6439980268478394, 0.0, 0.0, 0.0]  
"""     iou, iou_2d = box3d_iou(box1,box2,with_translation=True)
    print(iou)
    print(iou_2d) """
