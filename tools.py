import numpy as np
from numba import njit, int32, int64, double

@njit(int64(int64, int64, double[:], double))
def binary_search(imin, Nx, x, xi):
    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx - 2]:
        return Nx - 2

    # b. binary search
    half = Nx // 2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx // 2

    return imin

@njit(double(double[:], double[:], double[:], double[:], double[:], double[:], double[:,:,:,:,:,:], double, double, double, double, double, double, int32, int32, int32, int32, int32, int32), fastmath=True)
def _interp_6d(grid1, grid2, grid3, grid4, grid5, grid6, value, xi1, xi2, xi3, xi4, xi5, xi6, j1, j2, j3, j4, j5, j6):
    """ 6d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        grid5 (numpy.ndarray): 1d grid
        grid6 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (6d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point
        xi5 (double): input point
        xi6 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
        j3 (int): location in grid
        j4 (int): location in grid
        j5 (int): location in grid
        j6 (int): location in grid

    Returns:

        yi (double): output

    """
    
    # a. left/right
    nom_1_left = grid1[j1 + 1] - xi1
    nom_1_right = xi1 - grid1[j1]

    nom_2_left = grid2[j2 + 1] - xi2
    nom_2_right = xi2 - grid2[j2]

    nom_3_left = grid3[j3 + 1] - xi3
    nom_3_right = xi3 - grid3[j3]

    nom_4_left = grid4[j4 + 1] - xi4
    nom_4_right = xi4 - grid4[j4]

    nom_5_left = grid5[j5 + 1] - xi5
    nom_5_right = xi5 - grid5[j5]

    nom_6_left = grid6[j6 + 1] - xi6
    nom_6_right = xi6 - grid6[j6]

    # b. interpolation
    denom = (grid1[j1 + 1] - grid1[j1]) * (grid2[j2 + 1] - grid2[j2]) * (grid3[j3 + 1] - grid3[j3]) * \
            (grid4[j4 + 1] - grid4[j4]) * (grid5[j5 + 1] - grid5[j5]) * (grid6[j6 + 1] - grid6[j6])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right
            for k3 in range(2):
                nom_3 = nom_3_left if k3 == 0 else nom_3_right
                for k4 in range(2):
                    nom_4 = nom_4_left if k4 == 0 else nom_4_right
                    for k5 in range(2):
                        nom_5 = nom_5_left if k5 == 0 else nom_5_right
                        for k6 in range(2):
                            nom_6 = nom_6_left if k6 == 0 else nom_6_right
                            nom += nom_1 * nom_2 * nom_3 * nom_4 * nom_5 * nom_6 * value[j1 + k1, j2 + k2, j3 + k3, j4 + k4, j5 + k5, j6 + k6]

    return nom / denom

@njit(double(double[:], double[:], double[:], double[:], double[:], double[:], double[:,:,:,:,:,:], double, double, double, double, double, double), fastmath=True)
def interp_6d(grid1, grid2, grid3, grid4, grid5, grid6, value, xi1, xi2, xi3, xi4, xi5, xi6):
    """ 6d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        grid5 (numpy.ndarray): 1d grid
        grid6 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (6d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point
        xi5 (double): input point
        xi6 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0, grid1.size, grid1, xi1)
    j2 = binary_search(0, grid2.size, grid2, xi2)
    j3 = binary_search(0, grid3.size, grid3, xi3)
    j4 = binary_search(0, grid4.size, grid4, xi4)
    j5 = binary_search(0, grid5.size, grid5, xi5)
    j6 = binary_search(0, grid6.size, grid6, xi6)

    return _interp_6d(grid1, grid2, grid3, grid4, grid5, grid6, value, xi1, xi2, xi3, xi4, xi5, xi6, j1, j2, j3, j4, j5, j6)

@njit(double(double[:], double[:], double[:], double[:], double[:], double[:], double[:], double[:,:,:,:,:,:,:], double, double, double, double, double, double, double, int32, int32, int32, int32, int32, int32, int32), fastmath=True)
def _interp_7d(grid1, grid2, grid3, grid4, grid5, grid6, grid7, value, xi1, xi2, xi3, xi4, xi5, xi6, xi7, j1, j2, j3, j4, j5, j6, j7):
    """ 7d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        grid5 (numpy.ndarray): 1d grid
        grid6 (numpy.ndarray): 1d grid
        grid7 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (7d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point
        xi5 (double): input point
        xi6 (double): input point
        xi7 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
        j3 (int): location in grid
        j4 (int): location in grid
        j5 (int): location in grid
        j6 (int): location in grid
        j7 (int): location in grid

    Returns:

        yi (double): output

    """
    # a. left/right
    nom_1_left = grid1[j1 + 1] - xi1
    nom_1_right = xi1 - grid1[j1]

    nom_2_left = grid2[j2 + 1] - xi2
    nom_2_right = xi2 - grid2[j2]

    nom_3_left = grid3[j3 + 1] - xi3
    nom_3_right = xi3 - grid3[j3]

    nom_4_left = grid4[j4 + 1] - xi4
    nom_4_right = xi4 - grid4[j4]

    nom_5_left = grid5[j5 + 1] - xi5
    nom_5_right = xi5 - grid5[j5]

    nom_6_left = grid6[j6 + 1] - xi6
    nom_6_right = xi6 - grid6[j6]

    nom_7_left = grid7[j7 + 1] - xi7
    nom_7_right = xi7 - grid7[j7]

    # b. interpolation
    denom = (grid1[j1 + 1] - grid1[j1]) * (grid2[j2 + 1] - grid2[j2]) * (grid3[j3 + 1] - grid3[j3]) * \
            (grid4[j4 + 1] - grid4[j4]) * (grid5[j5 + 1] - grid5[j5]) * (grid6[j6 + 1] - grid6[j6]) * (grid7[j7 + 1] - grid7[j7])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right
            for k3 in range(2):
                nom_3 = nom_3_left if k3 == 0 else nom_3_right
                for k4 in range(2):
                    nom_4 = nom_4_left if k4 == 0 else nom_4_right
                    for k5 in range(2):
                        nom_5 = nom_5_left if k5 == 0 else nom_5_right
                        for k6 in range(2):
                            nom_6 = nom_6_left if k6 == 0 else nom_6_right
                            for k7 in range(2):
                                nom_7 = nom_7_left if k7 == 0 else nom_7_right
                                nom += nom_1 * nom_2 * nom_3 * nom_4 * nom_5 * nom_6 * nom_7 * value[j1 + k1, j2 + k2, j3 + k3, j4 + k4, j5 + k5, j6 + k6, j7 + k7]

    return nom / denom

@njit(double(double[:], double[:], double[:], double[:], double[:], double[:], double[:], double[:,:,:,:,:,:,:], double, double, double, double, double, double, double), fastmath=True)
def interp_7d(grid1, grid2, grid3, grid4, grid5, grid6, grid7, value, xi1, xi2, xi3, xi4, xi5, xi6, xi7):
    """ 7d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        grid5 (numpy.ndarray): 1d grid
        grid6 (numpy.ndarray): 1d grid
        grid7 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (7d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point
        xi5 (double): input point
        xi6 (double): input point
        xi7 (double): input point

    Returns:

        yi (double): output

    """
    # a. search in each dimension
    j1 = binary_search(0, grid1.size, grid1, xi1)
    j2 = binary_search(0, grid2.size, grid2, xi2)
    j3 = binary_search(0, grid3.size, grid3, xi3)
    j4 = binary_search(0, grid4.size, grid4, xi4)
    j5 = binary_search(0, grid5.size, grid5, xi5)
    j6 = binary_search(0, grid6.size, grid6, xi6)
    j7 = binary_search(0, grid7.size, grid7, xi7)

    return _interp_7d(grid1, grid2, grid3, grid4, grid5, grid6, grid7, value, xi1, xi2, xi3, xi4, xi5, xi6, xi7, j1, j2, j3, j4, j5, j6, j7)
