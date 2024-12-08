# %% Setup
import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

mi.set_variant('cuda_ad_rgb')

#hdr_format = 'exr'
hdr_format = 'png'

output_dir_base = 'inverse-rendering'
os.makedirs(output_dir_base, exist_ok=True)

# %% Load config
class Config():
    def __init__(self, scene_name, path, key,
                 render_spp=512,
                 spp_forward=32,
                 restir_spp=1,
                 mitsuba_spp=None,
                 grad_passes=1,
                 time=30e3,
                 lr=0.1,
                 lr_factor=0.5,
                 lr_updates=0,
                 restir_mcap=16,
                 max_depth=3,
                 include_restir_dr=False
    ):
        self.scene_name = scene_name
        self.path = path
        self.key = key
        self.render_spp = render_spp
        self.spp_forward = spp_forward
        self.restir_spp = restir_spp
        self.mitsuba_spp = restir_spp if mitsuba_spp is None else mitsuba_spp
        self.grad_passes = grad_passes
        self.time = time
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_updates = lr_updates
        self.restir_mcap = restir_mcap
        self.max_depth = max_depth
        self.include_restir_dr = include_restir_dr

scene_configs = {
    'veach-ajar': Config('veach-ajar', '../scenes/veach-ajar/scene.xml', 'LandscapeBSDF.brdf_0.reflectance.data',
        render_spp=512*8,
        spp_forward=32,
        restir_spp=1,
        time=30e3,
        lr=0.1,
        lr_factor=0.5,
        lr_updates=3,
        restir_mcap=16,
        max_depth=9,
        include_restir_dr=False,
    ),
    'living-room-2': Config('living-room-2', '../scenes/living-room-2/tire_living_room.xml', 'mat-tire.brdf_0.roughness.data',
        time=15e3,
        lr=0.01,
        max_depth=3,
        include_restir_dr=False,
    ),
    'staircase2': Config('staircase2', '../scenes/staircase2/scene.xml', 'FloorTilesBSDF.brdf_0.base_color.data',
        restir_spp=1,
        time=10e3,
        lr=0.05,
        restir_mcap=16,
        max_depth=2,
        include_restir_dr=True,
    ),
    'cornell-box': Config('cornell-box', '../scenes/cornell_box_picture/scene_picture.xml', 'PaintingBSDF.brdf_0.reflectance.data',
        restir_spp=1,
        time=30e3,
        lr=0.1,
        restir_mcap=16,
        max_depth=2,
        include_restir_dr=True,
    ),
}
config = scene_configs['cornell-box']

output_dir = os.path.join(output_dir_base, config.scene_name)
os.makedirs(output_dir, exist_ok=True)

# %% Utility functions
def write_image(name, data, hdr=True):
    mi.util.write_bitmap(os.path.join(output_dir, f'{name}.{hdr_format if hdr else "png"}'), data)

def convert_to_lum(grad_tensor, extend_dim=False):
    if len(grad_tensor.shape) != 3:
        return grad_tensor
    if grad_tensor.shape[2] == 1:
        if extend_dim:
            return grad_tensor
        return grad_tensor[:,:,0]
    grad_color = dr.unravel(mi.Color3f, dr.ravel(grad_tensor[...,:3]))
    grad_lum = mi.luminance(grad_color)
    shape = (grad_tensor.shape[0], grad_tensor.shape[1], 1) if extend_dim else \
        (grad_tensor.shape[0], grad_tensor.shape[1])
    return mi.TensorXf(grad_lum, shape=shape)

# Does multiple passes and combines the result if spp is large
def render_clean_image(scene, spp=config.render_spp):
    assert spp > 0
    max_spp = 512
    count = 1
    image = mi.render(scene=scene, seed=count, spp=min(spp, max_spp))
    while spp > max_spp:
        spp -= max_spp
        image += mi.render(scene=scene, seed=count, spp=min(spp, max_spp))
        count += 1
    return image / count

def set_max_depth(scene, max_depth=config.max_depth):
    scene.integrator().max_depth = max_depth
    # Disable russian roulette for now
    scene.integrator().rr_depth = max_depth + 10

def get_elapsed_execution_time():
    hist = dr.kernel_history()
    elapsed_time = 0
    for entry in hist:
        elapsed_time += entry['execution_time']
    return elapsed_time

def relse(a, b):
    return dr.sqr(a - b) / (dr.sqr(b) + 1e-2)

def relmse(a, b):
    return dr.mean(relse(a, b))

def mae(a, b):
    return dr.mean(dr.abs(a - b))

def derivative_err(img, ref):
    return dr.sum(relse(img, ref)) / dr.count(dr.neq(ref.array, 0))

def loss_func(image, image_gt):
    return relmse(image, image_gt)


# %% Load scene and parameters
print(f'-------------------- Running {config.scene_name} -------------------------')

scene = mi.load_file(config.path, integrator='restir_gi_dr')
set_max_depth(scene)

params = mi.traverse(scene)
print(params)
param_ref = mi.TensorXf(params[config.key])
param_shape = np.array(params[config.key].shape)
param_initial = np.full(param_shape.tolist(), 0.5)
if param_shape[2] == 4:
    param_initial[:,:,3] = 1
    param_ref[:,:,3] = 1

image_gt = render_clean_image(scene)
write_image('render_gt', image_gt, hdr=True)
write_image('texture_gt', param_ref, hdr=False)

params[config.key] = mi.TensorXf(param_initial)
params.update();
image_initial = render_clean_image(scene)
write_image('render_initial', image_initial, hdr=True)
write_image('texture_initial', param_initial, hdr=False)

opt = mi.ad.Adam(lr=config.lr)
opt[config.key] = params[config.key]
params.update(opt);
scene.integrator().param_name = config.key

dr.set_flag(dr.JitFlag.KernelHistory, 1)

# %% Run equal time optimization
def get_equal_time_optimization(use_ref, spp_grad):
    np.random.seed(0)
    # reset dr history
    get_elapsed_execution_time()

    # Reset initial params
    opt.reset(config.key)
    opt.set_learning_rate({'data': config.lr, config.key: config.lr})
    opt[config.key] = mi.TensorXf(param_initial)
    params.update(opt);
    scene.integrator().use_ref = use_ref
    scene.integrator().use_positivization = True
    scene.integrator().enable_temporal_reuse = True
    scene.integrator().M_cap = config.restir_mcap * spp_grad
    scene.integrator().reset()
    it = 0
    total_time = 0
    times = []
    losses = []
    errors = []

    progress_bar = tqdm(total=round(config.time), desc="Progress", position=0)
    update_lr_freq = 100 // (config.lr_updates + 1)
    lr_last_updated = 0
    while True:
        grads = dr.grad(opt[config.key])
        accumulated_loss = 0
        for _ in range(config.grad_passes):
            # Perform a (noisy) differentiable rendering of the scene
            image = mi.render(scene, params, spp=config.spp_forward,
                spp_grad=spp_grad,
                seed=np.random.randint(2**31))

            # Evaluate the objective function from the current rendered image
            loss = loss_func(image, image_gt)

            # Backpropagate through the rendering process
            dr.backward(loss)
            accumulated_loss += loss
            grads += dr.grad(opt[config.key])
            dr.set_grad(opt[config.key], 0)
        accumulated_loss /= config.grad_passes
        dr.set_grad(opt[config.key], grads / config.grad_passes)

        # Optimizer: take a gradient descent step
        opt.step()

        # Post-process the optimized parameters to ensure legal texture values.
        opt[config.key] = dr.clamp(opt[config.key], 0.001, 0.999)

        # Update the scene state to the new optimized values
        params.update(opt)
        total_time += get_elapsed_execution_time()

        percentage_time = total_time / config.time * 100
        if percentage_time >= lr_last_updated + update_lr_freq:
            lr_last_updated = percentage_time // update_lr_freq * update_lr_freq
            new_lr = config.lr_factor * opt.lr['data']
            opt.set_learning_rate({'data': new_lr, config.key: new_lr})

        with dr.suspend_grad():
            losses.append(accumulated_loss[0])
            errors.append(relmse(params[config.key], param_ref)[0])
        times.append(total_time / 1e3)
        progress_bar.set_postfix({"Error": f"{errors[-1]:.4f}", "Loss": f"{losses[-1]:.4f}", "Iteration": it+1, "lr": opt.lr['data']}, refresh=True)
        progress_bar.n = round(min(config.time, total_time))
        progress_bar.last_print_n = progress_bar.n
        progress_bar.update(0)

        if total_time > config.time:
            break

        it += 1

    progress_bar.close()

    return times, losses, errors, mi.TensorXf(params[config.key])

mitsuba_times, mitsuba_losses, mitsuba_errors, mitsuba_param = \
    get_equal_time_optimization(True, config.mitsuba_spp)

restir_times, restir_losses, restir_errors, restir_param = \
    get_equal_time_optimization(False, config.restir_spp)

if config.include_restir_dr:
    set_max_depth(scene, 2)
    restir_dr_times, restir_dr_losses, restir_dr_errors, restir_dr_param = \
        get_equal_time_optimization(False, config.restir_spp)
    set_max_depth(scene)

# %% Output equal time optimization
def plot_graph(ylabel, restir_values, mitsuba_values, restir_dr_values):
    plt.clf()
    plt.figure(figsize=(10, 4), dpi=100, constrained_layout=True);
    plt.plot(restir_times, restir_values, 'c-o', label='Ours', linewidth=6.0, markersize=4.0, mfc='white')
    plt.plot(mitsuba_times, mitsuba_values, 'm-o', label='Mitsuba 3', linewidth=6.0, markersize=4.0, mfc='white')
    if config.include_restir_dr:
        plt.plot(restir_dr_times, restir_dr_values, 'y-o', label='ReSTIR DR', linewidth=6.0, markersize=4.0, mfc='white')
    plt.xlabel('Time (s)');
    plt.ylabel(ylabel);
    plt.yscale('log')
    plt.legend();
    plt.savefig(os.path.join(output_dir, f'{ylabel}.pdf'), bbox_inches='tight', pad_inches=0.0)
plot_graph('Loss', restir_losses, mitsuba_losses, restir_dr_losses if config.include_restir_dr else None)
plot_graph('Error', restir_errors, mitsuba_errors, restir_dr_errors if config.include_restir_dr else None)

# %% Output equal time final image
params[config.key] = restir_param
params.update();
restir_image = render_clean_image(scene);
write_image('render_final_restir', restir_image, hdr=True)
write_image('texture_final_restir', restir_param, hdr=False)

params[config.key] = mitsuba_param
params.update();
mitsuba_image = render_clean_image(scene);
write_image('render_final_mitsuba', mitsuba_image, hdr=True)
write_image('texture_final_mitsuba', mitsuba_param, hdr=False)

if config.include_restir_dr:
    params[config.key] = restir_dr_param
    params.update();
    restir_dr_image = render_clean_image(scene);
    write_image('render_final_restir_dr', restir_dr_image, hdr=True)
    write_image('texture_final_restir_dr', restir_dr_param, hdr=False)

    # Render ReSTIR DR result with direct lighting only
    #set_max_depth(scene, 2)
    #restir_dr_image_direct = render_clean_image(scene);
    #write_image('render_final_restir_dr_direct', restir_dr_image_direct, hdr=True)
    #set_max_depth(scene)
