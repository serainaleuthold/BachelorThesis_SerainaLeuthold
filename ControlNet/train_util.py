import sys
sys.path.append("../data_gen")

import ceviche
from ceviche import fdfd_hz
from ceviche.constants import C_0
from stratton_chu_ceviche_rectangle import strattonChu2D

import torch
import numpy as np
import matplotlib.pyplot as plt
import io


plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 20  # Axis labels font size
plt.rcParams['xtick.labelsize'] = 20 # X tick labels font size
plt.rcParams['ytick.labelsize'] = 20 # Y tick labels font size
plt.rcParams['legend.fontsize'] = 18 # Legend font size

def get_intensity_farfield(farfield):
    return farfield[::-1,0]*farfield[::-1,0]+farfield[::-1,1]*farfield[::-1,1]

def plot_row(eps, Ex, theta_obs, farfield, rows, row, color='blue', vm_real=None, vm_imag=None, vm_intensity=None, with_power=False, normalize_intensity=True):
    cols = 3 if len(farfield.shape)==1 else 4 if not with_power else 5

    if eps is not None:
        plt.subplot(rows, cols, (row-1)*cols+1)
        plt.imshow(eps)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        plt.colorbar()
        #plt.title("dielectric")

    if Ex is not None:
        plt.subplot(rows, cols, (row-1)*cols+2)
        plt.imshow(Ex.real)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        plt.colorbar()
        #plt.title("Ex_real")

    if len(farfield.shape)==1:
        plt.subplot(rows, cols, (row-1)*cols+3, projection='polar')
        plt.plot(theta_obs, farfield[::-1], color=color)
        #plt.ylim((-vm_real,vm_real))
        #plt.title(f"farfield")
    else: # real and imag channels
        plt.subplot(rows, cols, (row-1)*cols+3, projection='polar')
        plt.plot(theta_obs, farfield[::-1,0], color=color)
        plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        plt.tick_params(axis='x', pad=15)
        #plt.ylim((-vm_real,vm_real))
        #plt.title(f"farfield_real")
        plt.subplot(rows, cols, (row-1)*cols+4, projection='polar')
        plt.plot(theta_obs, farfield[::-1,1], color=color)
        plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        plt.tick_params(axis='x', pad=15)
        #plt.ylim((-vm_imag,vm_imag))
        #plt.title(f"farfield_imag")

        if with_power:
            plt.subplot(rows, cols, (row-1)*cols+5, projection='polar')
            intensity = farfield[::-1,0]*farfield[::-1,0]+farfield[::-1,1]*farfield[::-1,1]
            if normalize_intensity:
                max_intensity = np.max(intensity)
                intensity = intensity/max_intensity
            plt.plot(theta_obs, intensity, color=color)
            plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
            plt.tick_params(axis='x', pad=15)
            #plt.ylim((0,vm_intensity))
            #plt.title(f"farfield_intensity")

def plot_small_row(eps, theta_obs, farfield, rows, row, color = 'blue', with_power=True, normalize_intensity=True):
    cols = 2 if len(farfield.shape)==1 else 3 if not with_power else 4

    if eps is not None:
        plt.subplot(rows, cols, (row-1)*cols+1)
        plt.imshow(eps)
        plt.colorbar()
        plt.title("dielectric")

    if len(farfield.shape)==1:
        plt.subplot(rows, cols, (row-1)*cols+2, projection='polar')
        plt.plot(theta_obs, farfield[::-1])
        #plt.ylim((-vm_real,vm_real))
        plt.title(f"farfield")
    else: # real and imag channels
        plt.subplot(rows, cols, (row-1)*cols+2, projection='polar')
        plt.plot(theta_obs, farfield[::-1,0])
        #plt.ylim((-vm_real,vm_real))
        plt.title(f"farfield_real")
        plt.subplot(rows, cols, (row-1)*cols+3, projection='polar')
        plt.plot(theta_obs, farfield[::-1,1])
        #plt.ylim((-vm_imag,vm_imag))
        plt.title(f"farfield_imag")

        if with_power:
            plt.subplot(rows, cols, (row-1)*cols+4, projection='polar')
            intensity = farfield[::-1,0]*farfield[::-1,0]+farfield[::-1,1]*farfield[::-1,1]
            if normalize_intensity:
                max_intensity = np.max(intensity)
                intensity = intensity/max_intensity
            plt.plot(theta_obs, intensity, color = color)
            #plt.ylim((0,vm_intensity))
            plt.title(f"farfield_intensity")

def analyse_device(struct):
    struct = struct.squeeze()
    num_ridges = 0
    h, w = struct.shape[0], struct.shape[1]
    assert h < w
    i=0
    isinridge = False
    ridge_shapes = []
    ridge_positions = []
    ridge_width=0
    for i in range(w):
        if struct[0][i] > 0.5:
            if not isinridge:
                ridge_positions.append(i)
                num_ridges += 1
                isinridge = True
                ridge_width += 1
            else:
                ridge_width += 1
        elif struct[0][i] <= 0.5:
            if isinridge:
                isinridge = False
                ridge_shapes.append(ridge_width)
                ridge_width = 0
        if i == (w-1):
            if isinridge:
                ridge_shapes.append(ridge_width)
            
    min_ridge, max_ridge = np.min(ridge_shapes), np.max(ridge_shapes)
    assert len(ridge_shapes) == num_ridges

    return ridge_positions, ridge_shapes


def thicken_near_box(data, th=5):
    for i in range(th):
        data[i+1,:] = data[i,:]
        data[-1-i,:] = data[-1,:]
        data[:,i+1] = data[:,i]
        data[:,-1-i] = data[:,-1]
    return data

def plot_row_box(eps, Ex, nearbox, rows, row, cols=3):
    plt.subplot(rows, cols, (row-1)*cols+1)
    plt.imshow(eps)
    plt.colorbar()
    plt.title("dielectric")

    plt.subplot(rows, cols, (row-1)*cols+2)
    plt.imshow(Ex.real)
    plt.colorbar()
    plt.title("Ex_real")

    plt.subplot(rows, cols, (row-1)*cols+3)
    plt.imshow(thicken_near_box(nearbox))
    plt.colorbar()
    plt.title("nearbox")

# for input is farfield intensity
#def process_farfield(farfield):
#    if isinstance(farfield, torch.Tensor):
#        result = torch.abs(farfield) * torch.abs(farfield) / 1000
#    elif isinstance(farfield, np.ndarray):
#        result = np.abs(farfield) * np.abs(farfield) / 1000
#    else:
#        raise ValueError("farfield type wrong")
#    return result

# for input is imaginary and real part of farfield


def process_farfield(farfield):
    if isinstance(farfield, torch.Tensor):
        if farfield.dtype in (torch.complex64, torch.complex128):
            result = torch.stack((farfield.real, farfield.imag), dim=-1) / 50
        else:
            print("farfield type: ", farfield.dtype)
            raise ValueError("farfield type wrong")
    elif isinstance(farfield, np.ndarray):
        if farfield.dtype in (np.complex64, np.complex128):
            result = np.stack((farfield.real, farfield.imag), axis=-1) / 50
        else:
            print("farfield type: ", farfield.dtype)
            raise ValueError("farfield type wrong")
    else:
        raise ValueError("farfield type wrong")
    return result

def process_nearbox(nearbox):
    # nearbox shape: (bs, 2, 96, 4)
    if isinstance(nearbox, torch.Tensor):
        result = torch.zeros((nearbox.shape[0], nearbox.shape[1], nearbox.shape[2], nearbox.shape[2]))
    elif isinstance(nearbox, np.ndarray):
        result = np.zeros((nearbox.shape[0], nearbox.shape[1], nearbox.shape[2], nearbox.shape[2]))
    result[:,:,0,:] = nearbox[:,:,:,0]
    result[:,:,-1,:] = nearbox[:,:,:,1]
    result[:,:,:,0] = nearbox[:,:,:,2]
    result[:,:,:,-1] = nearbox[:,:,:,3]

    return result

def sim_near_field(eps_r, spacing, args, ny=None, source_in=False):
    wavelength = args.wl
    omega = 2 * np.pi * C_0 / wavelength
    k0 = 2 * np.pi / wavelength

    F = fdfd_hz(omega, args.dL*1e-9, eps_r, [args.pml_x, args.pml_y])
    
    source_amp = 64e9/args.dL**2
    random_source = np.zeros((args.Nx, args.Ny), dtype=complex)

    if source_in:
    	random_source[args.pml_x+spacing-4, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = source_amp
    	random_source[args.pml_x+spacing-5, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = -source_amp*np.exp(-1j*k0*args.dL*1e-9)
    else:
    	random_source[args.pml_x+spacing, :] = source_amp

    Ex_forward, Ey_forward, Hz_forward = F.solve(random_source)
    
    return eps_r, Hz_forward, Ex_forward, Ey_forward, random_source

def norm(a):
	return np.sum(a*a)**.5
def simmilarity_score(ori_far, far):
    return np.sum(ori_far*far)/norm(ori_far)/norm(far)

def similarity_score_intensity(ori_far, far):
    intensity_ori = get_intensity_farfield(ori_far)
    intensity = get_intensity_farfield(far)
    return np.sum(intensity_ori*intensity)/norm(intensity_ori)/norm(intensity)

def simmilarity_score_without_mean(ori_far, far, far_mean):
    far = far - far_mean
    ori_far = ori_far - far_mean
    return np.sum(ori_far*far)/norm(ori_far)/norm(far)

def simmilarity_score_nearbox(ori_near, near):
    ori_near = np.concatenate([ori_near[0,:,0,:], ori_near[0,:,-1,:], ori_near[0,:,:,0], ori_near[0,:,:,-1]])
    near = np.concatenate([near[0,:,0,:], near[0,:,-1,:], near[0,:,:,0], near[0,:,:,-1]])
    return np.sum(ori_near*near)/norm(ori_near)/norm(near)

def frobenius(a, b):
    frob = np.trace(np.dot(np.conjugate(np.transpose(a)), b), axis1 = 0, axis2 = 1)
    assert np.isscalar(frob), 'frobenius product is not a scalar.'
    return frob

def similarity_structure(ori_eps, eps):
    ori_eps = ori_eps.squeeze()
    eps = eps.squeeze()
    assert ori_eps.ndim == 2, 'ori_eps has too many dim!'
    assert eps.ndim ==2, 'eps has too many dim!'
    norm_ori_eps = np.sqrt(frobenius(ori_eps, ori_eps))
    norm_eps = np.sqrt(frobenius(eps, eps))
    if norm_ori_eps < 1e-7 or norm_eps < 1e-7:
        print("no similarity")
        return 0
    normed_ori_eps = ori_eps/norm_ori_eps
    normed_eps = eps/norm_eps
    similarity = frobenius(normed_ori_eps, normed_eps)
    return similarity

def analyze_one_sample(step, args, ori_eps_array, original_farfield, generated_eps, component="Ex"):
    plot_N_generated = generated_eps.shape[0]

    eps_r = np.ones((args.Nx, args.Ny))
    spacing = int((args.Nx-2*args.pml_x-args.image_size)/3)
    if spacing < 20:
        raise ValueError("spacing between source, structure and PML is too small. Make simulation domain larger")

    nx, ny = args.image_size, args.image_size
    xc = (-args.Nx/2 + args.pml_x+2*spacing + int(nx/2))*args.dL*1e-9
    yc = 0
    Rx = int(nx/2 + 1*spacing/2)*args.dL*1e-9
    Ry = int(ny/2 + 1*spacing/2)*args.dL*1e-9
    sx = args.Nx*args.dL*1e-9
    sy = args.Ny*args.dL*1e-9

    ori_eps_array = ori_eps_array * (args.n_mat - 1) + 1
    eps_r[args.pml_x+2*spacing:args.pml_x+2*spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = ori_eps_array
    eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args)

    theta_obs, Far_Ex, Far_Ey, Far_Hz = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)

    farfield = Far_Ex if component=="Ex" else Far_Ey
    ori_farfield = process_farfield(farfield)

    plt.figure()
    plt.plot(original_farfield[0])
    plt.plot(ori_farfield)
    plt.savefig(args.model_saving_path+'/'+"debug_should_be_the_same.png")
    plt.close()

    plt.figure(figsize=(18,18))
    plot_field = Ex if component=="Ex" else Ey
    plot_row(eps.T, plot_field.T, theta_obs, ori_farfield, plot_N_generated+1, 1)

    similarities = []
    for i in range(plot_N_generated):
        gen_eps_array = generated_eps[i] * (args.n_mat - 1) + 1 # turn in to 1 and n_mat

        eps_r[args.pml_x+2*spacing:args.pml_x+2*spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = gen_eps_array
        eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args)
        theta_obs, Far_Ex, Far_Ey, Far_Hz = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)
        farfield = Far_Ex if component=="Ex" else Far_Ey
        farfield = process_farfield(farfield)
        plot_field = Ex if component=="Ex" else Ey
        plot_row(eps.T, plot_field.T, theta_obs, farfield, plot_N_generated+1, i+2)
        similarities.append(simmilarity_score(farfield, ori_farfield))

    
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png')
    buffer.seek(0)
    image = buffer.getvalue()
    plt.savefig(args.model_saving_path+'/'+f"sample_step{step}.png", dpi=300, transparent=True)
    plt.close()

    return np.mean(similarities)

def analyze_one_sample_source_in(step, args, ori_eps_array, original_farfield, generated_eps, component="Ey", name="debug"):
    plot_N_generated = generated_eps.shape[0]

    eps_r = np.ones((args.Nx, args.Ny))

    if hasattr(args, 'image_size'):
        image_size = args.image_size
        nx, ny = args.image_size, args.image_size
    else:
        image_size = max(args.image_sizex, args.image_sizey)
        nx, ny = args.image_sizex, args.image_sizey

    spacing = int((args.Nx-2*args.pml_x-image_size)/2)
    if spacing < 20:
        raise ValueError("spacing between source, structure and PML is too small. Make simulation domain larger")

    xc = (-args.Nx/2 + args.pml_x+1*spacing + int(image_size/2))*args.dL*1e-9
    yc = 0
    Rx = int(image_size/2 + 1*spacing/2)*args.dL*1e-9
    Ry = int(image_size/2 + 1*spacing/2)*args.dL*1e-9
    sx = args.Nx*args.dL*1e-9
    sy = args.Ny*args.dL*1e-9

    ori_eps_array = ori_eps_array * (args.n_mat - 1) + 1
    eps_r[args.pml_x+spacing:args.pml_x+spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = ori_eps_array
    eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args, source_in=True, ny=ny)

    theta_obs, Far_Ex, Far_Ey, Far_Hz = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)

    farfield = Far_Ex if component=="Ex" else Far_Ey
    ori_farfield = process_farfield(farfield)

    plt.figure()
    plt.plot(original_farfield[0,:,0])
    plt.plot(ori_farfield[:,0])
    plt.savefig(args.model_saving_path+'/'+"debug_should_be_the_same.png")
    plt.close()

    plt.figure(figsize=(18,18))
    plot_field = Ex if component=="Ex" else Ey

    vm_real = 1.2*np.max(np.abs(ori_farfield[:,0]))
    vm_imag = 1.2*np.max(np.abs(ori_farfield[:,1]))
    vm_intensity = 1.2*np.max(np.abs(ori_farfield[::-1,0]*ori_farfield[::-1,0]+ori_farfield[::-1,1]*ori_farfield[::-1,1]))
    plot_row(eps.T, plot_field.T, theta_obs, ori_farfield, plot_N_generated+1, 1, vm_real=vm_real, vm_imag=vm_imag, vm_intensity=vm_intensity, with_power=True)

    similarities = []
    for i in range(plot_N_generated):
        gen_eps_array = generated_eps[i] * (args.n_mat - 1) + 1 # turn in to 1 and n_mat

        eps_r[args.pml_x+1*spacing:args.pml_x+1*spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = gen_eps_array
        eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args, source_in=True, ny=ny)
        theta_obs, Far_Ex, Far_Ey, Far_Hz = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)
        farfield = Far_Ex if component=="Ex" else Far_Ey
        farfield = process_farfield(farfield)
        plot_field = Ex if component=="Ex" else Ey
        plot_row(eps.T, plot_field.T, theta_obs, farfield, plot_N_generated+1, i+2, vm_real=vm_real, vm_imag=vm_imag, vm_intensity=vm_intensity, with_power=True)
        similarities.append(simmilarity_score(farfield, ori_farfield))

    canvas = plt.gcf().canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype = 'uint8').reshape((h, w, 3))

    plt.savefig(args.model_saving_path+'/samples/farfields/'+f"sample_step{step}.png", dpi=300, transparent=True)
    plt.close()

    return np.mean(similarities), image

def analyze_inverse_design_source_in(step, args, original_farfield, generated_eps, component="Ey", name="inv"):
    plot_N_generated = generated_eps.shape[0]
    eps_r = np.ones((args.Nx, args.Ny))

    if hasattr(args, 'image_size'):
        image_size = args.image_size
        nx, ny = args.image_size, args.image_size
    else:
        image_size = max(args.image_sizex, args.image_sizey)
        nx, ny = args.image_sizex, args.image_sizey

    spacing = int((args.Nx-2*args.pml_x-image_size)/2)
    if spacing < 20:
        raise ValueError("spacing between source, structure and PML is too small. Make simulation domain larger")

    xc = (-args.Nx/2 + args.pml_x+1*spacing + int(image_size/2))*args.dL*1e-9
    yc = 0
    Rx = int(image_size/2 + 1*spacing/2)*args.dL*1e-9
    Ry = int(image_size/2 + 1*spacing/2)*args.dL*1e-9
    sx = args.Nx*args.dL*1e-9
    sy = args.Ny*args.dL*1e-9

    plt.figure(figsize=(18,18))
    plot_row(None, None, np.linspace(0,2*np.pi, num=original_farfield.shape[0]), original_farfield, plot_N_generated+1, 1, with_power=True)

    similarities = []
    for i in range(plot_N_generated):
        gen_eps_array = generated_eps[i] * (args.n_mat - 1) + 1 # turn in to 1 and n_mat

        eps_r[args.pml_x+1*spacing:args.pml_x+1*spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = gen_eps_array
        eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args, source_in=True, ny=ny)
        theta_obs, Far_Ex, Far_Ey, Far_Hz = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)
        farfield = Far_Ex if component=="Ex" else Far_Ey
        farfield = process_farfield(farfield)
        plot_field = Ex if component=="Ex" else Ey
        plot_row(eps.T, plot_field.T, theta_obs, farfield, plot_N_generated+1, i+2, with_power=True)
        similarities.append(simmilarity_score(farfield, original_farfield))

    plt.savefig(args.model_saving_path+'/'+f"sample_step{step}_{name}.png", dpi=300, transparent=True)
    plt.close()
    return np.mean(similarities)

def analyze_one_sample_near_box(step, args, ori_eps_array, original_nearbox, generated_eps, component="Ex"):
    plot_N_generated = generated_eps.shape[0]

    eps_r = np.ones((args.Nx, args.Ny))
    spacing = int((args.Nx-2*args.pml_x-args.image_size)/2)
    if spacing < 20:
        raise ValueError("spacing between source, structure and PML is too small. Make simulation domain larger")

    nx, ny = args.image_size, args.image_size
    xc = (-args.Nx/2 + args.pml_x+1*spacing + int(nx/2))*args.dL*1e-9
    yc = 0
    Rx = int(nx/2 + 1*spacing/2)*args.dL*1e-9
    Ry = int(ny/2 + 1*spacing/2)*args.dL*1e-9
    sx = args.Nx*args.dL*1e-9
    sy = args.Ny*args.dL*1e-9

    ori_eps_array = ori_eps_array * (args.n_mat - 1) + 1
    eps_r[args.pml_x+spacing:args.pml_x+spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = ori_eps_array
    eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args, source_in=True, ny=ny)

    # theta_obs, Far_Ex, Far_Ey, Far_Hz = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)

    x_start = args.pml_x+int(1/2*spacing)
    x_end = args.pml_x+int(1/2*spacing) + nx + spacing

    y_start = int(args.Ny/2)-int(ny/2 + 1*spacing/2)
    y_end = int(args.Ny/2)-int(ny/2 + 1*spacing/2) + ny + spacing

    if component=="Ex":
        nearbox = np.stack([Ex[None,x_start, y_start:y_end], Ex[None,x_end-1, y_start:y_end], Ex[None,x_start:x_end, y_start], Ex[None,x_start:x_end, y_end-1]], axis=-1)
        nearbox = np.stack((nearbox.real, nearbox.imag), axis=1)
    else:
        nearbox = np.stack([Ey[None,x_start, y_start:y_end], Ey[None,x_end-1, y_start:y_end], Ey[None,x_start:x_end, y_start], Ey[None,x_start:x_end, y_end-1]], axis=-1)
        nearbox = np.stack((nearbox.real, nearbox.imag), axis=1)
    
    ori_nearbox = process_nearbox(nearbox)

    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(thicken_near_box(original_nearbox[0,0]))
    plt.subplot(2,1,2)
    plt.imshow(thicken_near_box(ori_nearbox[0,0]))
    plt.savefig(args.model_saving_path+'/'+"debug_should_be_the_same.png")
    plt.close()

    plt.figure(figsize=(18,18))
    plot_field = Ex if component=="Ex" else Ey
    plot_row_box(eps.T, plot_field.T, ori_nearbox[0,0].T, plot_N_generated+1, 1)

    similarities = []
    for i in range(plot_N_generated):
        gen_eps_array = generated_eps[i] * (args.n_mat - 1) + 1 # turn in to 1 and n_mat

        eps_r[args.pml_x+1*spacing:args.pml_x+1*spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = gen_eps_array
        eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args, source_in=True, ny=ny)
        # theta_obs, Far_Ex, Far_Ey, Far_Hz = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)
        if component=="Ex":
            nearbox = np.stack([Ex[None,x_start, y_start:y_end], Ex[None,x_end-1, y_start:y_end], Ex[None,x_start:x_end, y_start], Ex[None,x_start:x_end, y_end-1]], axis=-1)
            nearbox = np.stack((nearbox.real, nearbox.imag), axis=1)
        else:
            nearbox = np.stack([Ey[None,x_start, y_start:y_end], Ey[None,x_end-1, y_start:y_end], Ey[None,x_start:x_end, y_start], Ey[None,x_start:x_end, y_end-1]], axis=-1)
            nearbox = np.stack((nearbox.real, nearbox.imag), axis=1)

        nearbox = process_nearbox(nearbox)
        plot_field = Ex if component=="Ex" else Ey
        plot_row_box(eps.T, plot_field.T, nearbox[0,0].T, plot_N_generated+1, i+2)
        similarities.append(simmilarity_score_nearbox(nearbox, ori_nearbox))
    
    canvas = plt.gcf().canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype = 'uint8').reshape((h, w, 3))

    plt.savefig(args.model_saving_path+'/'+f"sample_step{step}.png", dpi=300, transparent=True)
    plt.close()

    return np.mean(similarities), image 


def test_interpolation_old(step, args, sample_eps, input_farfield, ori_eps_array, ori_farfield_array, component = 'Ey'):
    N_generated = args.num_parallel_samples

    if hasattr(args, 'image_size'):
        image_size = args.image_size
        nx, ny = args.image_size, args.image_size
    else:
        image_size = max(args.image_sizex, args.image_sizey)
        nx, ny = args.image_sizex, args.image_sizey

    spacing = int((args.Nx-2*args.pml_x-image_size)/2)
    if spacing < 20:
        raise ValueError("spacing between source, structure and PML is too small. Make simulation domain larger")

    xc = (-args.Nx/2 + args.pml_x+1*spacing + int(image_size/2))*args.dL*1e-9
    yc = 0
    Rx = int(image_size/2 + 1*spacing/2)*args.dL*1e-9
    Ry = int(image_size/2 + 1*spacing/2)*args.dL*1e-9
    sx = args.Nx*args.dL*1e-9
    sy = args.Ny*args.dL*1e-9

    sample_eps = sample_eps * (args.n_mat - 1) + 1
    sample_farfields = []
    theta_obs=0

    for i in range(N_generated):
        eps_r = np.ones((args.Nx, args.Ny))
        eps_r[args.pml_x+spacing:args.pml_x+spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = sample_eps[i,0,:,:]
        eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args, source_in=True, ny=ny)
        theta_obs, Far_Ex, Far_Ey, Far_Hz = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)
        farfield = Far_Ex if component=="Ex" else Far_Ey
        sample_farfields.append(process_farfield(farfield))

    similarity_in_out = 0
    sample_farfield = np.empty_like(sample_farfields[0])
    for i in range(N_generated):
        similarity_in_out_loop = simmilarity_score(input_farfield, sample_farfields[i])
        if similarity_in_out_loop > similarity_in_out:
            similarity_in_out = similarity_in_out_loop
            sample_farfield = sample_farfields[i]
    assert sample_farfield.any(), "sample_farfield is empty"

    top_three_sim = [0, 0, 0]
    indices = [0, 0, 0]
    for i in range(0, ori_farfield_array.shape[0]):
        min_sim = min(top_three_sim)
        min_index = top_three_sim.index(min_sim)
        similarity = simmilarity_score(sample_farfield, ori_farfield_array[i])
        if any(sim == similarity for sim in top_three_sim):
            print("Has double structures")
        if similarity > min_sim:
            top_three_sim[min_index] = similarity
            indices[min_index] = i
    top_three_eps = [ori_eps_array[i] for i in indices]
    top_three_farfield = [ori_farfield_array[i] for i in indices]

    plt.figure(figsize=(30, 30))
    plot_field = Ex if component=="Ex" else Ey

    vm_real = 1.2*np.max(np.abs(input_farfield[:,0]))
    vm_imag = 1.2*np.max(np.abs(input_farfield[:,1]))
    vm_intensity = 1.2*np.max(np.abs(input_farfield[::-1,0]*input_farfield[::-1,0]+input_farfield[::-1,1]*input_farfield[::-1,1]))


    plot_row(None, None, theta_obs, input_farfield, len(top_three_sim)+2, 1, vm_real=vm_real, vm_imag=vm_imag, vm_intensity=vm_intensity, with_power=True)
    plot_field = Ey
    plot_row(eps.T, plot_field, theta_obs, sample_farfield, len(top_three_sim)+2, 2, vm_real=vm_real, vm_imag=vm_imag, vm_intensity=vm_intensity, with_power=True)
    plt.text(1, 0, f'Similarity Score Farfields (input vs. output): {similarity_in_out}', ha='right', va='bottom', transform=plt.gca().transAxes)
    print("Similarity in vs out: ", similarity_in_out)

    top_three_sim_struct = [0, 0, 0]
    top_three_sim_input = [0, 0, 0]
    for i in range(len(top_three_sim)):
        orig_eps = top_three_eps[i] * (args.n_mat - 1) + 1 # turn in to 1 and n_mat

        eps_r = np.ones((args.Nx, args.Ny))
        eps_r[args.pml_x+1*spacing:args.pml_x+1*spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = orig_eps
        ori_eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args, source_in=True, ny=ny)
        theta_obs, Far_Ex_ori, Far_Ey_ori, Far_Hz_ori = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)
        Far_Ey_ori = process_farfield(Far_Ey_ori)
        top_three_sim_struct[i] = similarity_structure(ori_eps, eps)
        top_three_sim_input[i] = simmilarity_score(Far_Ey_ori, input_farfield)
        plot_field = Ex if component=="Ex" else Ey
        plot_row(ori_eps.T, plot_field, theta_obs, Far_Ey_ori, len(top_three_sim)+2, i+3, vm_real=vm_real, vm_imag=vm_imag, vm_intensity=vm_intensity, with_power=True)
        plt.text(1, 0.9, f'Similarity Score Structures: {top_three_sim_struct[i]}', ha='right', va='bottom', transform=plt.gca().transAxes)
        plt.text(1, 0.7, f'Similarity Score Farfield to Sample: {top_three_sim[i]}', ha='right', va='bottom', transform=plt.gca().transAxes)
        plt.text(1, 0.5, f'Similarity Score Farfield to Input: {top_three_sim_input[i]}', ha='right', va='bottom', transform=plt.gca().transAxes)

    canvas = plt.gcf().canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype = 'uint8').reshape((h, w, 3))

    plt.savefig(args.save_dir_samples+'testing_interpolation'+'/'+f"test_interpolation_{step}.png", dpi=300, transparent=True)
    plt.close()

    top_three_sim_max = np.max(top_three_sim)
    top_three_sim_struct_max = top_three_sim_struct[top_three_sim.index(top_three_sim_max)]
    top_three_sim_input_max = top_three_sim_input[top_three_sim.index(top_three_sim_max)]


    return top_three_sim_max, top_three_sim_input_max, top_three_sim_struct_max, similarity_in_out, image


def evaluate_top_n(array, comparison_arrays, compare_fn, top_n=4):
    top_n_values = [0]*top_n
    indices = [0]*top_n
    top_n_arrays = np.empty((top_n,) + array.shape)
    if isinstance(comparison_arrays, list):
        data_length = len(comparison_arrays)
    else:
        data_length = comparison_arrays.shape[0]
    for i in  range(data_length):
        min_value = min(top_n_values)
        if isinstance(top_n_values.index(min_value), list):
            print("Is a list: ", top_n_values.index(min_value))
        elif not isinstance(top_n_values.index(min_value), int):
            print("is multiple values: ", top_n_values.index(min_value))
        curr_value = compare_fn(array, comparison_arrays[i])
        #if any(np.array_equal(curr_value, x) for x in top_n_values):
        #    continue

        if curr_value > min_value:
            indices[top_n_values.index(min_value)] = i
            top_n_arrays[top_n_values.index(min_value)] = comparison_arrays[i]
            top_n_values[top_n_values.index(min_value)] = curr_value
    
    return top_n_values, top_n_arrays, indices

def get_farfield(args, sample_eps, component = 'Ey'):
    if hasattr(args, 'image_size'):
        image_size = args.image_size
        nx, ny = args.image_size, args.image_size
    else:
        image_size = max(args.image_sizex, args.image_sizey)
        nx, ny = args.image_sizex, args.image_sizey

    spacing = int((args.Nx-2*args.pml_x-image_size)/2)
    if spacing < 20:
        raise ValueError("spacing between source, structure and PML is too small. Make simulation domain larger")

    xc = (-args.Nx/2 + args.pml_x+1*spacing + int(image_size/2))*args.dL*1e-9
    yc = 0
    Rx = int(image_size/2 + 1*spacing/2)*args.dL*1e-9
    Ry = int(image_size/2 + 1*spacing/2)*args.dL*1e-9
    sx = args.Nx*args.dL*1e-9
    sy = args.Ny*args.dL*1e-9

    sample_eps = sample_eps * (args.n_mat - 1) + 1

    eps_r = np.ones((args.Nx, args.Ny))
    eps_r[args.pml_x+spacing:args.pml_x+spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = sample_eps[0,:,:]
    eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args, source_in=True, ny=ny)
    theta_obs, Far_Ex, Far_Ey, Far_Hz = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)
    farfield = Far_Ex if component=="Ex" else Far_Ey
    nearfield = Ex if component=="Ex" else Ey
    farfield = process_farfield(farfield)

    return farfield, nearfield, theta_obs


def test_interpolation(step, args, sample_eps, input_farfield, input_struct, ori_eps_array, ori_farfield_array, num_devices, handcrafted=False, only_best_in_one_plot=False, component = 'Ey'):

    N_generated = args.num_parallel_samples

    if hasattr(args, 'image_size'):
        image_size = args.image_size
        nx, ny = args.image_size, args.image_size
    else:
        image_size = max(args.image_sizex, args.image_sizey)
        nx, ny = args.image_sizex, args.image_sizey

    spacing = int((args.Nx-2*args.pml_x-image_size)/2)
    if spacing < 20:
        raise ValueError("spacing between source, structure and PML is too small. Make simulation domain larger")

    xc = (-args.Nx/2 + args.pml_x+1*spacing + int(image_size/2))*args.dL*1e-9
    yc = 0
    Rx = int(image_size/2 + 1*spacing/2)*args.dL*1e-9
    Ry = int(image_size/2 + 1*spacing/2)*args.dL*1e-9
    sx = args.Nx*args.dL*1e-9
    sy = args.Ny*args.dL*1e-9

    sample_eps = sample_eps * (args.n_mat - 1) + 1
    sample_farfields = []
    theta_obs=0

    for i in range(N_generated):
        eps_r = np.ones((args.Nx, args.Ny))
        eps_r[args.pml_x+spacing:args.pml_x+spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = sample_eps[i,0,:,:]
        eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args, source_in=True, ny=ny)
        theta_obs, Far_Ex, Far_Ey, Far_Hz = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)
        farfield = Far_Ex if component=="Ex" else Far_Ey
        sample_farfields.append(process_farfield(farfield))


    top_similarity_in_out, top_sample_farfields, indices_sample = evaluate_top_n(input_farfield, np.array(sample_farfields), similarity_score_intensity, top_n = num_devices)
    top_similarity_in_ori, top_original_farfields, indices_original = evaluate_top_n(input_farfield, ori_farfield_array, similarity_score_intensity, top_n = num_devices)

    top_ori_farfield_structs = ori_eps_array[indices_original]
    top_sample_farfield_structs = ori_eps_array[indices_sample]

    if not handcrafted:
        top_sim_struct_in_out, top_sample_struct, _ = evaluate_top_n(input_struct, np.array(sample_eps), similarity_structure, top_n = num_devices)
        top_sim_struct_in_ori, top_original_struct, _ = evaluate_top_n(input_struct, ori_eps_array, similarity_structure, top_n = num_devices)
    
    if handcrafted:
        input_stuct = None
    else:
        input_struct = input_struct.T

    
    max_index_in_out = top_similarity_in_out.index(np.max(top_similarity_in_out))
    max_index_in_ori = top_similarity_in_ori.index(np.max(top_similarity_in_ori))
    max_farfield_ori = top_original_farfields[max_index_in_ori]
    max_farfield_out = top_sample_farfields[max_index_in_out]

    best_generated_struct_positions, best_generated_struct_shape = analyse_device(top_sample_farfield_structs[max_index_in_out])
    best_original_struct_positions, best_original_struct_shape = analyse_device(top_ori_farfield_structs[max_index_in_ori])

    if only_best_in_one_plot:
        plt.figure(figsize=(30, 30))
        plot_small_row(input_struct, theta_obs, input_farfield, 4, 1, color = 'dodgerblue', with_power = True)
        plot_small_row(None, theta_obs, max_farfield_ori, 4, 1, color = 'orange', with_power = True)
        plot_small_row(None, theta_obs, max_farfield_out, 4, 1, color = 'yellowgreen', with_power = True)
        plot_small_row(input_struct, theta_obs, input_farfield, 4, 2, color = 'dodgerblue', with_power = True)
        plot_small_row(top_ori_farfield_structs[max_index_in_ori].T, theta_obs, max_farfield_ori, 4, 3, color = 'orange', with_power = True)
        plot_small_row(top_sample_farfield_structs[max_index_in_out].T, theta_obs, max_farfield_out, 4, 4, color = 'yellowgreen', with_power = True)
        print("in plots, max in out, max in ori: ", np.max(top_similarity_in_out), np.max(top_similarity_in_ori))

        plt.tight_layout()
        plt.savefig(args.save_dir_samples+'testing_interpolation'+'/'+f"test_interpolation_{step}_only_best.png", dpi=300, transparent=True)
        plt.close()

    else:
        plt.figure(figsize=(30, 30))
        plot_small_row(input_struct, theta_obs, input_farfield, num_devices + 1, 1, with_power = True)
        for i in range(num_devices):
            plot_small_row(top_ori_farfield_structs[i].T, theta_obs, top_original_farfields[i], num_devices + 1, i+2, with_power = True)

        plt.tight_layout()
        plt.savefig(args.save_dir_samples+'testing_interpolation'+'/'+f"test_interpolation_{step}_ori_vs_in.png", dpi=300, transparent=True)
        plt.close()

        plt.figure(figsize=(30, 30))
        plot_small_row(input_struct, theta_obs, input_farfield, num_devices + 1, 1, with_power = True)
        for i in range(num_devices):
            plot_small_row(top_sample_farfield_structs[i].T, theta_obs, top_sample_farfields[i], num_devices + 1, i+2, with_power = True)

        plt.tight_layout()
        plt.savefig(args.save_dir_samples+'testing_interpolation'+'/'+f"test_interpolation_{step}_gen_vs_in.png", dpi=300, transparent=True)
        plt.close()

    max_sim_farfield_in_ori = np.max(top_similarity_in_ori)
    average_sim_farfield_in_out = np.mean(top_similarity_in_out)
    max_sim_farfield_in_out = np.max(top_similarity_in_out)
    max_sim_farfield_index = top_similarity_in_out.index(np.max(top_similarity_in_out))


    if not handcrafted:
        max_sim_struct_in_ori = np.max(top_sim_struct_in_ori)
        average_sim_struct_in_out = np.mean(top_sim_struct_in_out)   

    if not handcrafted:
        return (max_sim_farfield_in_ori, 
                average_sim_farfield_in_out, 
                max_sim_farfield_in_out, 
                max_sim_farfield_index, 
                max_sim_struct_in_ori, 
                average_sim_struct_in_out,
                best_generated_struct_positions, 
                best_generated_struct_shape,
                best_original_struct_positions, 
                best_original_struct_shape)
    else:
        return (max_sim_farfield_in_ori, 
                average_sim_farfield_in_out, 
                max_sim_farfield_in_out, 
                max_sim_farfield_index,
                best_generated_struct_positions, 
                best_generated_struct_shape,
                best_original_struct_positions, 
                best_original_struct_shape)



    """

        plot_row(None, None, theta_obs, input_farfield, 2*num_devices+2, 1, vm_real=vm_real, vm_imag=vm_imag, vm_intensity=vm_intensity, with_power=True)
        plot_field = Ey
        plot_row(eps.T, plot_field, theta_obs, sample_farfield, 2*num_devices+2, 2, vm_real=vm_real, vm_imag=vm_imag, vm_intensity=vm_intensity, with_power=True)


        for i in range(num_devices):
            orig_eps = top_ori_farfield_structs[i] * (args.n_mat - 1) + 1 # turn in to 1 and n_mat

            eps_r = np.ones((args.Nx, args.Ny))
            eps_r[args.pml_x+1*spacing:args.pml_x+1*spacing+nx, int(args.Ny/2-ny/2):int(args.Ny/2-ny/2)+ny] = orig_eps
            ori_eps, Hz, Ex, Ey, source = sim_near_field(eps_r, spacing, args, source_in=True, ny=ny)
            theta_obs, Far_Ex_ori, Far_Ey_ori, Far_Hz_ori = strattonChu2D(0, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.wl, Ex.T, Ey.T, -Hz.T, N_theta=args.N_theta, debug=0)
            Far_Ey_ori = process_farfield(Far_Ey_ori)
            top_three_sim_struct[i] = similarity_structure(ori_eps, eps)
            top_three_sim_input[i] = simmilarity_score(Far_Ey_ori, input_farfield)
            plot_field = Ex if component=="Ex" else Ey
            plot_row(ori_eps.T, plot_field, theta_obs, Far_Ey_ori, len(top_three_sim)+2, i+3, vm_real=vm_real, vm_imag=vm_imag, vm_intensity=vm_intensity, with_power=True)
            plt.text(1, 0.9, f'Similarity Score Structures: {top_three_sim_struct[i]}', ha='right', va='bottom', transform=plt.gca().transAxes)
            plt.text(1, 0.7, f'Similarity Score Farfield to Sample: {top_three_sim[i]}', ha='right', va='bottom', transform=plt.gca().transAxes)
            plt.text(1, 0.5, f'Similarity Score Farfield to Input: {top_three_sim_input[i]}', ha='right', va='bottom', transform=plt.gca().transAxes)


    canvas = plt.gcf().canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype = 'uint8').reshape((h, w, 3))
    """