import torch
import utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
    
stage1_model = 'ipirm'
num_shot = 10
fp = 'misc/env_ref_set_vipriors{}_rn50_{}_pretrained'.format(num_shot, stage1_model)
env_ref_set = torch.load(fp)
data = './data/DataSets/CMNIST_wcolor/64'
memory_images = utils.Imagenet_idx(root=data + '/train', transform=None, target_transform=None)
num_samples = len(memory_images.imgs)
all_idx = list(range(num_samples))

for k, indeces in env_ref_set.items():
    fig, ax = plt.subplots(1, 2, figsize=(2*5, 4))
    env0_n, env1_n = indeces[0].tolist(), indeces[1].tolist()
    env_a = list(set(all_idx) - set(env0_n) - set(env1_n))
    
    i = 0
    col_env0, col_env1 = [memory_images.imgs[j][1] // 2 for j in env0_n], [memory_images.imgs[j][1] // 2 for j in env1_n]
    
    total_0 = len(col_env0)
    total_1 = len(col_env1)
    counts_0 = [col_env0.count(0), col_env0.count(1)]
    counts_1 = [col_env1.count(0), col_env1.count(1)]
    perc_0 = [count / total_0 * 100 for count in counts_0]
    perc_1 = [count / total_1 * 100 for count in counts_1]
    
    labels = ['R', 'G']
    x = np.arange(len(labels))
    width = 0.35
    colors_hatches = ['red', 'lime']
    bar = ax[i].bar(x - width/2, perc_0, width, label='env_0', hatch="x", color='lightsteelblue', kwargs={"hatch_linewidth": 3.})
    """
    for j, bc in enumerate(bar):
        bc._hatch_color = mpl.colors.to_rgba(colors_hatches[j])
        bc.stale = True
    """
    
    bar = ax[i].bar(x + width/2, perc_1, width, label='env_1', hatch="x", color='orange', kwargs={"hatch_linewidth": 3.})
    """
    for j, bc in enumerate(bar):
        bc._hatch_color = mpl.colors.to_rgba(colors[j])
        bc.stale = True
    """
        
    ax[i].set_ylabel('Percentage (%)')
    ax[i].set_xlabel('Color')
    ax[i].set_title(f'Percentage of colors R/G in envs 0/1 for anchor {k}')
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(labels)
    ax[i].legend()
    
    i += 1
    col_a = [memory_images.imgs[j][1] // 2 for j in env_a]

    total_a = len(col_a)
    counts_a = [col_a.count(0), col_a.count(1)]
    perc_a = [count / total_a * 100 for count in counts_a]
    perc_tot = [(counts_a[i] + counts_0[i] + counts_1[i]) / (total_a + total_0 + total_1) * 100 for i in range(len(counts_a)) ]
    
    labels = ['R', 'G']
    x = np.arange(len(labels))
    width = 0.35

    colors = ['red', 'green']
    ax[i].bar(x - width/2, perc_a, width, label='anchor', color=colors)
    colors = ['orangered', 'limegreen']
    ax[i].bar(x + width/2, perc_tot, width, label='total', color=colors)
    
    ax[i].set_ylabel('Percentage (%)')
    ax[i].set_xlabel('Color')
    ax[i].set_title(f'Percentage of colors R/G in for anchor {k}')
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(labels)
    ax[i].legend()  
    
    plt.show()

    idxs = env0_n + env_a
    col0 = [memory_images.imgs[j][1] // 2 for j in idxs]
    tar0 = [memory_images.imgs[j][1] % 2 for j in idxs]
    corr0 = np.corrcoef(np.array(col0), np.array(tar0))
    idxs = env1_n + env_a
    col1 = [memory_images.imgs[j][1] // 2 for j in idxs]
    tar1 = [memory_images.imgs[j][1] % 2 for j in idxs]
    corr1 = np.corrcoef(np.array(col1), np.array(tar1))
    print("Color/Label correlations:", "env 0:", corr0[0,1], "env 1:", corr1[0,1])