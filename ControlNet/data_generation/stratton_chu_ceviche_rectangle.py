import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def plotNearField(idx, out_folder, sx, sy, X_surf, Y_surf, X_glob, Y_glob, field_OBJ, field_surf, component_string):
    plt.figure()
    # Plot the original field as a heatmap
    plt.imshow(np.real(field_OBJ), extent=[-sx/2, sx/2, -sy/2, sy/2], cmap='viridis', aspect='equal', interpolation='nearest', origin='lower')
    plt.colorbar(label=component_string+' Field Magnitude')

    # Plot the ellipse
    plt.plot(X_surf, Y_surf, 'k-', label='Ellipse Outline')

    # Plot field along the ellipse
    plt.scatter(X_surf, Y_surf, c=np.real(field_surf), cmap='viridis', label=component_string+'_surf on Ellipse')
    plt.scatter(np.min(X_glob),np.min(Y_glob),c='r', label='Bottom Left Corner')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Interpolated '+component_string+' along Ellipse over Original '+component_string+' Field')
    plt.legend()
    plt.axis('equal')
    # plt.show()
    plt.savefig(out_folder+ '/' + component_string+f'_{idx}.png')
    plt.close()

def strattonChu2D(idx, dL, sx, sy, Nx, Ny, xc, yc, Rx, Ry, lambda_val, Ex_OBJ, Ey_OBJ, Hz_OBJ, N_theta=361, debug=1, points_per_pixel=1, out_folder="."):
    # Coordinates of the rectangle's corners
    x_left = xc - Rx
    x_right = xc + Rx
    y_bottom = yc - Ry
    y_top = yc + Ry

    vertical_dist = y_top - y_bottom
    horizontal_dist = x_right - x_left

    npx_per_vertical_edge = round(points_per_pixel * vertical_dist/dL)
    npx_per_horizontal_edge = round(points_per_pixel * horizontal_dist/dL)

    # Generate points for each edge
    # Bottom edge (left to right)
    X_bottom = np.linspace(x_left, x_right, npx_per_horizontal_edge)
    Y_bottom = np.full(npx_per_horizontal_edge, y_bottom)

    # Right edge (bottom to top)
    Y_right = np.linspace(y_bottom, y_top, npx_per_vertical_edge)
    X_right = np.full(npx_per_vertical_edge, x_right)

    # Top edge (right to left)
    X_top = np.linspace(x_right, x_left, npx_per_horizontal_edge)
    Y_top = np.full(npx_per_horizontal_edge, y_top)

    # Left edge (top to bottom)
    Y_left = np.linspace(y_top, y_bottom, npx_per_vertical_edge)
    X_left = np.full(npx_per_vertical_edge, x_left)

    # Combine the points
    X_surf = np.concatenate([X_bottom, X_right, X_top, X_left])
    Y_surf = np.concatenate([Y_bottom, Y_right, Y_top, Y_left])

    y_linspace = np.linspace(-sy / 2, sy / 2, Ny)
    x_linspace = np.linspace(-sx / 2, sx / 2, Nx)
    X_glob, Y_glob = np.meshgrid(x_linspace, y_linspace, indexing='xy')

    # Flatten the global coordinates and the field for interpolation
    points = np.array([X_glob.flatten(), Y_glob.flatten()]).T
    ex_values = Ex_OBJ.flatten()
    ey_values = Ey_OBJ.flatten()
    hz_values = Hz_OBJ.flatten()

    # Define the ellipse coordinates as query points
    ellipse_points = np.array([X_surf, Y_surf]).T

    # Perform the interpolation
    if debug:
        print("Performing interpolation...")
    Ex_surf = griddata(points, ex_values, ellipse_points, method='nearest')
    Ey_surf = griddata(points, ey_values, ellipse_points, method='nearest')
    Hz_surf = griddata(points, hz_values, ellipse_points, method='nearest')
    if debug:
        print("Interpolation complete")

    if debug:
        plotNearField(idx, out_folder, sx, sy, X_surf, Y_surf, X_glob, Y_glob, Ex_OBJ, Ex_surf, 'Ex')
        plotNearField(idx, out_folder, sx, sy, X_surf, Y_surf, X_glob, Y_glob, Ey_OBJ, Ey_surf, 'Ey')
        plotNearField(idx, out_folder, sx, sy, X_surf, Y_surf, X_glob, Y_glob, Hz_OBJ, Hz_surf, 'Hz')

    # Define surfaces for use in numerical integrals
    #bottom
    nx_bottom = 0*np.ones(npx_per_horizontal_edge)
    ny_bottom = -1*np.ones(npx_per_horizontal_edge)

    #right
    nx_right = 1*np.ones(npx_per_vertical_edge)
    ny_right = 0*np.ones(npx_per_vertical_edge)

    #top
    nx_top = 0*np.ones(npx_per_horizontal_edge)
    ny_top = 1*np.ones(npx_per_horizontal_edge)

    #left
    nx_left = -1*np.ones(npx_per_vertical_edge)
    ny_left = 0*np.ones(npx_per_vertical_edge)

    nx = np.concatenate([nx_bottom, nx_right, nx_top, nx_left])
    ny = np.concatenate([ny_bottom, ny_right, ny_top, ny_left])

    if( (X_glob[0,1:-1]-X_glob[0,0:-2])[0] != (Y_glob[1:-1, 0]-Y_glob[0:-2, 0])[0] ):
        raise ValueError("dx and dy must be equal")
    dl_surf = (X_glob[0,1:-1]-X_glob[0,0:-2])[0]

    # Perform N2F for each point
    Z0 = 120 * np.pi  # Characteristic impedance of free space
    c = 299792458  # Speed of light in vacuum
    eps0 = 8.854187817e-12  # Permittivity of free space
    k0 = 2 * np.pi / lambda_val

    theta_obs = np.linspace(0, 2*np.pi, N_theta)
    r_obs = 10000 * lambda_val  # Assuming lambda_val is defined somewhere

    Far_Ex = np.zeros(len(theta_obs), dtype=np.complex128)
    Far_Ey = np.zeros(len(theta_obs), dtype=np.complex128)
    Far_Hz = np.zeros(len(theta_obs), dtype=np.complex128)

    for it_theta in range(len(theta_obs)):
        theta = theta_obs[it_theta]

        ux = np.cos(theta)
        uy = np.sin(theta)
        r_rs = np.sqrt((r_obs * ux - X_surf)**2 + (r_obs * uy - Y_surf)**2)
        phas_kr = np.exp(-1j * k0 * r_rs) / (4*np.pi*r_rs)

        Jx = Hz_surf * ny
        Jy = Hz_surf * nx
        Mz = -(Ey_surf * nx - Ex_surf * ny)
        Mux = Mz * uy
        Muy = Mz * ux
        rho = -(Ex_surf * nx + Ey_surf * ny)

        Far_Ex[it_theta] = np.sum( 1j*k0 * ( Z0*(ny*Hz_surf) - 
                                            (nx*Ey_surf-ny*Ex_surf)*uy + 
                                            (nx*Ex_surf+ny*Ey_surf)*ux ) * phas_kr * dl_surf)

        Far_Ey[it_theta] = np.sum( 1j*k0 * ( -Z0*(nx*Hz_surf) + 
                                            (nx*Ey_surf-ny*Ex_surf)*ux + 
                                            (nx*Ex_surf+ny*Ey_surf)*uy ) * phas_kr * dl_surf)          

        Far_Hz[it_theta] = - np.sum( 1j*k0 * ( c*eps0*(nx*Ey_surf-ny*Ex_surf) - 
                                               ( (ny*Hz_surf)*uy + (nx*Hz_surf)*ux ) ) * phas_kr * dl_surf)        
    if debug:
        print("Far field calculation complete (Stratton-Chu)")

    return theta_obs, Far_Ex, Far_Ey, Far_Hz
