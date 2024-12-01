# %%
import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

mi.set_variant('cuda_ad_rgb')

output_dir_base = 'inverse-rendering'
os.makedirs(output_dir_base, exist_ok=True)

# %% Load scene and parameters
scenes = [
    {
        'name': 'tire',
        'path': '../scenes/tire/scene.xml',
        'key': 'mat-tire.brdf_0.roughness.data',
        'restir_spp': 1,
        'mitsuba_spp': 1,
        'time': 90e3,
        'learning_rate': 0.005,
        'restir_mcap': 16,
        'integrator': 'restir_gi_dr',
        'max_depth': 2
    },
    {
        'name': 'ashtray',
        'path': '../scenes/ashtray/scene.xml',
        'key': 'mat-ashtray.brdf_0.anisotropic.data',
        'restir_spp': 1,
        'mitsuba_spp': 1,
        'time': 30e3,
        'learning_rate': 0.01,
        'restir_mcap': 16,
        'integrator': 'restir_gi_dr',
        'max_depth': 2
    },
    {
        'name': 'chalice',
        'path': '../scenes/chalice/scene.xml',
        'key': 'mat-chalice.brdf_0.roughness.data',
        'restir_spp': 1,
        'mitsuba_spp': 1,
        'time': 30e3,
        'learning_rate': 0.01,
        'restir_mcap': 16,
        'integrator': 'restir_gi_dr',
        'max_depth': 2
    },
    {
        'name': 'cbox',
        'path': '../scenes/cornell-box/scene.xml',
        'key': 'NewTallBoxBSDF.brdf_0.base_color.data',
        'restir_spp': 1,
        'mitsuba_spp': 1,
        'time': 30e3,
        'learning_rate': 0.1,
        'restir_mcap': 16,
        'integrator': 'restir_gi_dr',
        'max_depth': 3
    },
    {
        'name': 'staircase2',
        'path': '../scenes/staircase2/scene.xml',
        'key': 'FloorTilesBSDF.brdf_0.diffuse_reflectance.data',
        'restir_spp': 1,
        'mitsuba_spp': 1,
        'time': 30e3,
        'learning_rate': 0.1,
        'restir_mcap': 16,
        'integrator': 'restir_gi_dr',
        'max_depth': 2
    },
    {
        'name': 'veach-ajar',
        'path': '../scenes/veach-ajar/scene.xml',
        'key': 'LandscapeBSDF.brdf_0.reflectance.data',
        'restir_spp': 1,
        'mitsuba_spp': 1,
        'time': 60e3,
        'learning_rate': 0.1,
        'learning_rate_factor': 0.5,
        'learning_rate_update_times': 3,
        'restir_mcap': 64,
        'integrator': 'restir_gi_dr',
        'max_depth': 3
    },
]
scene_info = scenes[-1]

render_spp = 512*8
spp_forward = 32

scene_path, scene_name, key = scene_info['path'], scene_info['name'], scene_info['key']
output_dir = os.path.join(output_dir_base, scene_name)
os.makedirs(output_dir, exist_ok=True)

# %%
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

def render_clean_image(scene, spp=render_spp):
    assert spp > 0
    max_spp = 512
    count = 1
    image = mi.render(scene=scene, seed=count, spp=min(spp, max_spp))
    while spp > max_spp:
        spp -= max_spp
        image += mi.render(scene=scene, seed=count, spp=min(spp, max_spp))
        count += 1
    return image / count

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


# %%
print(f'-------------------- Running {scene_name} -------------------------')

scene = mi.load_file(scene_path, integrator=scene_info['integrator'])
scene.integrator().max_depth = scene_info['max_depth']
scene.integrator().rr_depth = 1000
is_base_color = 'base_color' in key
if 'learning_rate' in scene_info:
    learning_rate = scene_info['learning_rate']
elif is_base_color:
    learning_rate = 0.1
else:
    learning_rate = 0.01

image_gt = render_clean_image(scene)
mi.util.write_bitmap(os.path.join(output_dir, 'render_gt.exr'), image_gt)

params = mi.traverse(scene)
print(params)
param_ref = mi.TensorXf(params[key])
param_shape = np.array(params[key].shape)
mi.util.write_bitmap(os.path.join(output_dir, 'texture_gt.png'), param_ref)

param_initial = np.full(param_shape.tolist(), 0.5)
if param_shape[2] == 4:
    param_initial[:,:,3] = 1
    param_ref[:,:,3] = 1
params[key] = mi.TensorXf(param_initial)
mi.util.write_bitmap(os.path.join(output_dir, 'texture_initial.png'), param_initial)

params.update();

image_initial = render_clean_image(scene)
mi.util.write_bitmap(os.path.join(output_dir, 'render_initial.exr'), image_initial)

opt = mi.ad.Adam(lr=learning_rate)
opt[key] = params[key]
params.update(opt);
scene.integrator().param_name = key

dr.set_flag(dr.JitFlag.KernelHistory, 1)


params[key] = mi.TensorXf(param_initial)
params.update();

opt = mi.ad.Adam(lr=learning_rate)
opt[key] = params[key]
params.update(opt);

# %% Run equal time optimization
def get_equal_time_optimization(use_ref, n_time, spp_forward, spp_grad, M_cap=1):
    np.random.seed(0)
    # reset dr history
    get_elapsed_execution_time()

    # Reset initial params
    opt.reset(key)
    opt.set_learning_rate({'data': learning_rate, key: learning_rate})
    opt[key] = mi.TensorXf(param_initial)
    params.update(opt);
    scene.integrator().use_ref = use_ref
    scene.integrator().use_positivization = True
    scene.integrator().enable_temporal_reuse = True
    scene.integrator().M_cap = M_cap * spp_grad
    if scene_info['integrator'] != 'prb':
        scene.integrator().reset()
    it = 0
    total_time = 0
    times = []
    losses = []

    progress_bar = tqdm(total=round(n_time), desc="Progress", position=0)
    update_lr_freq = 100 // (scene_info.get('learning_rate_update_times', 4) + 1)
    lr_last_updated = 0
    lr_factor = scene_info.get('learning_rate_factor', 1)
    while True:
        grads = dr.grad(opt[key])
        for i in range(1):
            # Perform a (noisy) differentiable rendering of the scene
            mi.rende
            image = mi.render(scene, params, spp=spp_forward,
                spp_grad=spp_grad,
                seed=np.random.randint(2**31))

            # Evaluate the objective function from the current rendered image
            loss = loss_func(image, image_gt)

            # Backpropagate through the rendering process
            dr.backward(loss)
            grads += dr.grad(opt[key])
            dr.set_grad(opt[key], 0)
        dr.set_grad(opt[key], grads / 1)

        # Optimizer: take a gradient descent step
        opt.step()

        # Post-process the optimized parameters to ensure legal color values.
        opt[key] = dr.clamp(opt[key], 0.001, 0.999)

        # Update the scene state to the new optimized values
        params.update(opt)
        total_time += get_elapsed_execution_time()

        if lr_factor != 1:
            percentage_time = total_time / n_time * 100
            if percentage_time >= lr_last_updated + update_lr_freq:
                lr_last_updated = percentage_time // update_lr_freq * update_lr_freq
                new_lr = lr_factor * opt.lr['data']
                opt.set_learning_rate({'data': new_lr, key: new_lr})
        # losses.append(loss[0])
        with dr.suspend_grad():
            losses.append(relmse(params[key], param_ref)[0])
        times.append(total_time / 1e3)
        progress_bar.set_postfix({"Error": f"{losses[-1]:.4f}", "Iteration": it+1, "lr": opt.lr['data']}, refresh=True)
        progress_bar.n = round(min(n_time, total_time))
        progress_bar.last_print_n = progress_bar.n
        progress_bar.update(0)

        if total_time > n_time:
            break

        #print(f'-- Iteration {it} -- Loss {losses[-1]:.6f} --')
        it += 1

    progress_bar.close()

    return times, losses, mi.TensorXf(params[key])

restir_times, restir_losses, restir_param = \
    get_equal_time_optimization(False, scene_info['time'], spp_forward, scene_info['restir_spp'], M_cap=scene_info['restir_mcap'])

mitsuba_times, mitsuba_losses, mitsuba_param = \
    get_equal_time_optimization(True, scene_info['time'], spp_forward, scene_info['mitsuba_spp'])

# %% Output equal time optimization
plt.clf()
plt.figure(figsize=(10, 4), dpi=100, constrained_layout=True);
plt.plot(mitsuba_times, mitsuba_losses, 'm-o', label='Mitsuba 3', linewidth=6.0, markersize=4.0, mfc='white')
plt.plot(restir_times, restir_losses, 'c-o', label='Ours', linewidth=6.0, markersize=4.0, mfc='white')
plt.xlabel('Time (s)');
plt.ylabel('Error');
plt.yscale('log')
plt.legend();
plt.savefig(os.path.join(output_dir, 'inv_convergence.pdf'), bbox_inches='tight', pad_inches=0.0)

# %% Output equal time final image
params[key] = mitsuba_param
params.update();
mitsuba_image = render_clean_image(scene);
mi.util.write_bitmap(os.path.join(output_dir, 'render_final_mitsuba.exr'), mitsuba_image)
mi.util.write_bitmap(os.path.join(output_dir, 'texture_final_mitsuba.png'), mitsuba_param)

params[key] = restir_param
params.update();
restir_image = render_clean_image(scene);
mi.util.write_bitmap(os.path.join(output_dir, 'render_final_restir.exr'), restir_image)
mi.util.write_bitmap(os.path.join(output_dir, 'texture_final_restir.png'), restir_param)

restir_img_err = restir_losses[-1]
mitsuba_img_err = mitsuba_losses[-1]

print(
    f'ReSTIR, error: {restir_img_err:.6e} ({restir_img_err/mitsuba_img_err:.5f}x)\n'
    f'Mitsuba, error: {mitsuba_img_err:.6e} (1.00x)\n'
)

with open(os.path.join(output_dir, 'inv_convergence.json'), 'w') as f:
    json_str = json.dumps({
        'mitsuba_times': mitsuba_times,
        'mitsuba_losses': mitsuba_losses,
        'restir_times': restir_times,
        'restir_losses': restir_losses,
        'mitsuba_img_err': mitsuba_img_err,
        'restir_img_err': restir_img_err,
        'img_err_reduction': restir_img_err/mitsuba_img_err,
    }, indent=2)
    f.write(json_str)
