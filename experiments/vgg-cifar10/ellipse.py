def ellipsoid_grid_count ( n, r, c ):

#*****************************************************************************
#
## ellipsoid_grid_count() counts the grid points inside an ellipsoid.
#
#  Discussion:
#
#    The ellipsoid is specified as
#
#      ( ( X - C0 ) / R0 )^2 
#    + ( ( Y - C1 ) / R1 )^2 
#    + ( ( Z - C2 ) / R2 )^2 = 1
#
#    The user supplies a number N.  There will be N+1 grid points along
#    the shortest axis.
#
#  Input:
#
#    integer N, the number of subintervals.
#
#    real R[3], the half axis lengths.
#
#    real C[3], the center of the ellipsoid.
#
#  Output:
#
#    integer NG, the number of grid points inside the ellipsoid.
#
  if ( r[0] == min ( r ) ):
    h = 2.0 * r[0] / float ( 2 * n + 1 )
    ni = n
    nj = int ( 1.0 + r[1] / r[0] ) * n
    nk = int ( 1.0 + r[2] / r[0] ) * n
  elif ( r[1] == min ( r ) ):
    h = 2.0 * r[1] / float ( 2 * n + 1 )
    nj = n
    ni = int ( 1.0 + r[0] / r[1] ) * n
    nk = int ( 1.0 + r[2] / r[1] ) * n
  else:
    h = 2.0 * r[2] / float ( 2 * n + 1 )
    nk = n
    ni = int ( 1.0 + r[0] / r[2] ) * n
    nj = int ( 1.0 + r[1] / r[2] ) * n

  ng = 0

  for k in range ( 0, nk + 1 ):
    z = c[2] + float ( k ) * h
    for j in range ( 0, nj + 1 ):
      y = c[1] + float ( j ) * h
      for i in range ( 0, ni + 1 ):
        x = c[0] + float ( i ) * h
        if ( 1.0 < ( ( x - c[0] ) / r[0] ) ** 2 \
                 + ( ( y - c[1] ) / r[1] ) ** 2 \
                 + ( ( z - c[2] ) / r[2] ) ** 2 ):
          break
#
#  At least one point is generated, but more possible by symmetry.
#
        np = 1
        if ( 0 < i ):
          np = 2 * np
        if ( 0 < j ):
          np = 2 * np
        if ( 0 < k ):
          np = 2 * np

        ng = ng + np

  return ng


def ellipsoid_grid_display ( r, c, ng, xg, filename ):

#*****************************************************************************
#
## ellipsoid_grid_display() displays grid points inside a ellipsoid.
#
#  Input:
#
#    real R[3], the half axis lengths.
#
#    real C[3], the center of the ellipsoid.
#
#    integer NG, the number of grid points inside the ellipsoid.
#
#    real XYZ[NG,3], the grid point coordinates.
#
#    string FILENAME, the name of the plotfile to be created.
#
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  fig = plt.figure ( )
  ax = fig.add_subplot ( 111, projection = '3d' )
  ax.scatter ( xg[:,0], xg[:,1], xg[:,2], 'b' );

  ax.set_xlabel ( '<---X--->' )
  ax.set_ylabel ( '<---Y--->' )
  ax.set_zlabel ( '<---Z--->' )
  ax.set_title ( 'Grid points in ellipsoid' )
  ax.grid ( True )
# ax.axis ( 'equal' )
  plt.savefig ( filename )
  plt.show ( block = False )
  plt.close ( )

  return


def ellipsoid_grid_points ( n, r, c, ng ):

#*****************************************************************************80
#
## ellipsoid_grid_points() generates the grid points inside an ellipsoid.
#
#  Discussion:
#
#    The ellipsoid is specified as
#
#      ( ( X - C0 ) / R0 )^2 
#    + ( ( Y - C1 ) / R1 )^2 
#    + ( ( Z - C2 ) / R2 )^2 = 1
#
#    The user supplies a number N.  There will be N+1 grid points along
#    the shortest axis.
#
#  Input:
#
#    integer N, the number of subintervals.
#
#    real R[3], the half axis lengths.
#
#    real C[3], the center of the ellipsoid.
#
#    integer NG, the number of grid points inside the ellipsoid.
#
#  Output:
#
#    real XYZ[NG,3], the grid point coordinates.
#
  import numpy as np

  if ( r[0] == min ( r ) ):
    h = 2.0 * r[0] / float ( 2 * n + 1 )
    ni = n
    nj = int ( 1.0 + r[1] / r[0] ) * n
    nk = int ( 1.0 + r[2] / r[0] ) * n
  elif ( r[1] == min ( r ) ):
    h = 2.0 * r[1] / float ( 2 * n + 1 )
    nj = n
    ni = int ( 1.0 + r[0] / r[1] ) * n
    nk = int ( 1.0 + r[2] / r[1] ) * n
  else:
    h = 2.0 * r[2] / float ( 2 * n + 1 )
    nk = n
    ni = int ( 1.0 + r[0] / r[2] ) * n
    nj = int ( 1.0 + r[1] / r[2] ) * n

  xyz = np.zeros ( ( ng, 3 ) )

  p = np.zeros ( ( 8, 3 ) )

  ng2 = 0

  for k in range ( 0, nk + 1 ):
    z = c[2] + float ( k ) * h
    for j in range ( 0, nj + 1 ):
      y = c[1] + float ( j ) * h
      for i in range ( 0, ni + 1 ):
        x = c[0] + float ( i ) * h
#
#  If we have left the ellipsoid, the I loop is completed.
#
        if ( 1.0 < ( ( x - c[0] ) / r[0] ) ** 2 \
                 + ( ( y - c[1] ) / r[1] ) ** 2 \
                 + ( ( z - c[2] ) / r[2] ) ** 2 ):
          break
#
#  At least one point is generated, but more possible by symmetry.
#
        p[0,0] = x
        p[0,1] = y
        p[0,2] = z
        np = 1

        if ( 0 < i ):
          for l in range ( 0, np ):
            p[np+l,0] = 2.0 * c[0] - p[l,0]
            p[np+l,1] =              p[l,1]
            p[np+l,2] =              p[l,2]
          np = 2 * np

        if ( 0 < j ):
          for l in range ( 0, np ):
            p[np+l,0] =              p[l,0]
            p[np+l,1] = 2.0 * c[1] - p[l,1]
            p[np+l,2] =              p[l,2]
          np = 2 * np

        if ( 0 < k ):
          for l in range ( 0, np ):
            p[np+l,0] =              p[l,0]
            p[np+l,1] =              p[l,1]
            p[np+l,2] = 2.0 * c[2] - p[l,2]
          np = 2 * np

        for l in range ( 0, np ):
          xyz[ng2+l,0] = p[l,0]
          xyz[ng2+l,1] = p[l,1]
          xyz[ng2+l,2] = p[l,2]
        ng2 = ng2 + np

  return xyz