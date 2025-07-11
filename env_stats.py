import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import utils
import matplotlib as mpl
import numpy as np

import os
import argparse


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook (Kaggle too!)
        elif shell == 'TerminalInteractiveShell':
            return False  # IPython terminal
        else:
            return False
    except NameError:
        return False  # Standard script

def is_headless():
    if os.name != "nt":
        return os.environ.get("DISPLAY", "") == ""
    else:
        return False

def setup_backend():
    if is_notebook():
        print("Detected Jupyter notebook: using inline backend.")
        from IPython import get_ipython
        get_ipython().run_line_magic("matplotlib", "inline")
    elif os.name == "nt":
        try:
            mpl.use("TkAgg")
            print("Using TkAgg backend on Windows (popup windows).")
        except ImportError as e:
            print("Could not set TkAgg backend:", e)
            print("Falling back to default backend.")
    elif is_headless():
        mpl.use("Agg")
        print("Headless environment: using Agg backend (save to file).")
    else:
        print("Using default interactive backend.")

    import matplotlib.pyplot as plt
    return plt

def main(args):
    
    # --- Call once at the top of your script ---
    plt = setup_backend()

    mpl_version_str = mpl.__version__
    mpl_version_tuple = tuple(int(part) for part in mpl_version_str.split('.'))
    hatches_linewidth_supported = mpl_version_tuple >= (3, 10, 0)

    fp = 'misc/env_ref_set_vipriors{}_rn50_{}_pretrained'.format(args.num_shot, args.stage1_model)
    env_ref_set = torch.load(fp)
    data = './data/DataSets/CMNIST_wcolor/64'
    memory_images = utils.Imagenet_idx(root=data + '/train', transform=None, target_transform=None)
    num_samples = len(memory_images.imgs)
    all_idx = list(range(num_samples))

    """
    env_ref_set is a dictionary "over" class labels.
    each entry is a tuple over class-environments (K = 2) of sample indices in loader that are assigned to that environment (equal number)
    the environments have been precomputed by running the images through the default (pre-trained) model, calculated the (corrected) cosine
    distance between the "other" samples and anchor samples, sorting in descending order the distances and splitting the result 50/50 into
    two environments.
    """
    # memory_images are ALL images - anchor and other
    label = 1
    env0 = 0
    env1 = 1

    anchor = 0
    print(f'-----------ANCHOR {anchor}------------')
    # sum of numbers of samples in env0 + env1 vs number of "other" samples
    #print(1, f'anchor {anchor}:', 'env0 + env1:', len(env_ref_set[anchor][env0]) + len(env_ref_set[anchor][env1]), \
    #    'other:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 != anchor]))

    # number of samples with colors 0/1 in env0 and env1 and their sum
    count_c0_o = [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env0]].count(0) + \
                 [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env1]].count(0)
    count_c1_o = [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env0]].count(1) + \
                 [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env1]].count(1)

    # number of anchor samples
    indx_a = [j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 == anchor]
    # number of anchor samples with colors 0/1 and their sum
    count_c0_a = [memory_images.imgs[j][label] // 2 for j in indx_a].count(0)
    count_c1_a = [memory_images.imgs[j][label] // 2 for j in indx_a].count(1)

    # number of samples with colors 0/1 in env0 and env1 and their sum
    #print(2, f'anchor {anchor}:',f'env {env0}:', count_c0_o, f'env {env1}:', count_c1_o, \
    #    'total other by color:', count_c0_o + count_c1_o, \
    #    'total other by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 != anchor]))
    
    # number of anchor samples with colors 0/1 and their sum
    #print(3, f'anchor {anchor}:', count_c0_a, count_c1_a, \
    #    'total anchor by color:', count_c0_a + count_c1_a, \
    #    'total anchor by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 == anchor]))
    
    # total number of samples of different colors for anchor
    count_c0_a0 = count_c0_a + count_c0_o
    count_c1_a0 = count_c1_a + count_c1_o
    #print(4, 'Color 0:', count_c0_a0, 'Color 1:', count_c1_a0, 'Sum - # samples:', count_c0_a0 + count_c1_a0 - len(memory_images))

    anchor = 1
    print(f'-----------ANCHOR {anchor}------------')
    # sum of numbers of samples in env0 + env1 vs number of "other" samples
    #print(5, f'anchor {anchor}:', 'env0 + env1:', len(env_ref_set[anchor][env0]) + len(env_ref_set[anchor][env1]), \
    #    'other:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 != anchor]))

    # number of samples with colors 0/1 in env0 and env1 and their sum
    count_c0_o = [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env0]].count(0) + \
                 [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env1]].count(0)
    count_c1_o = [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env0]].count(1) + \
                 [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env1]].count(1)

    # number of anchor samples
    indx_a = [j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 == anchor]
    # number of anchor samples with colors 0/1 and their sum
    count_c0_a = [memory_images.imgs[j][label] // 2 for j in indx_a].count(0)
    count_c1_a = [memory_images.imgs[j][label] // 2 for j in indx_a].count(1)

    # number of samples with colors 0/1 in env0 and env1 and their sum
    #print(6, f'anchor {anchor}:',f'env {env0}:', count_c0_o, f'env {env1}:', count_c1_o, \
    #    'total other by color:', count_c0_o + count_c1_o, \
    #    'total other by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 != anchor]))

    # number of anchor samples with colors 0/1 and their sum
    #print(7, f'anchor {anchor}:', count_c0_a, count_c1_a, \
    #    'total anchor by color:', count_c0_a + count_c1_a, \
    #    'total anchor by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 == anchor]))
    
    # total number of samples of different colors for anchor
    count_c0_a1 = count_c0_a + count_c0_o
    count_c1_a1 = count_c1_a + count_c1_o
    #print(8, 'Color 0:', count_c0_a1, 'Color 1:', count_c1_a1, 'Sum - # samples:', count_c0_a1 + count_c1_a1 - len(memory_images))
    
    # total number of samples
    print(f'-----------Total number of samples------------')
    print(9, count_c0_a0 + count_c0_a1 + count_c1_a0 + count_c1_a1, 2*len(memory_images))

    return

    for k, indeces in env_ref_set.items():
        fig, ax = plt.subplots(1, 2, figsize=(2*5, 4))
        env0_n, env1_n = indeces[0].tolist(), indeces[1].tolist()
        env_a = list(set(all_idx) - set(env0_n) - set(env1_n))
        env0_n = [x for x in env0_n if x < num_samples]
        env1_n = [x for x in env1_n if x < num_samples]

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
        if hatches_linewidth_supported:
            bar = ax[i].bar(x - width/2, perc_0, width, label='env_0', hatch="x", color='lightsteelblue', hatch_linewidth=3.0)
        else:
            bar = ax[i].bar(x - width/2, perc_0, width, label='env_0', hatch="x", color='lightsteelblue')

        for j, bc in enumerate(bar):
            bc._hatch_color = mpl.colors.to_rgba(colors_hatches[j])
            bc.stale = True

        if hatches_linewidth_supported:
            bar = ax[i].bar(x + width/2, perc_1, width, label='env_1', hatch="x", color='orange', hatch_linewidth=3.)
        else:
            bar = ax[i].bar(x + width/2, perc_1, width, label='env_1', hatch="x", color='orange')

        for j, bc in enumerate(bar):
            bc._hatch_color = mpl.colors.to_rgba(colors_hatches[j])
            bc.stale = True

        ax[i].set_ylabel('Percentage (%)')
        ax[i].set_xlabel('Color')
        ax[i].set_title(f'Percentage of colors R/G in envs 0/1 for anchor {k}')
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels)
        ax[i].legend()
        ax[i].grid(True)

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
        ax[i].set_title(f'Percentage of colors R/G for anchor {k}')
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels)
        ax[i].legend()  
        ax[i].grid(True)

        # How to handle final display or saving
        if is_notebook():
            # In notebook, it automatically displays inline
            pass
        elif is_headless():
            plt.savefig(f"plot_{k}.png")
            plt.close()  # Close figure to avoid duplicate output and free memory
            print(f"Plot saved to plot{k}.png (headless mode).")
        else:
            plt.show(block = k == len(env_ref_set)-1)

        idxs = env0_n + env_a
        col0 = [memory_images.imgs[j][1] // 2 for j in idxs]
        tar0 = [memory_images.imgs[j][1] % 2 for j in idxs]
        corr0 = np.corrcoef(np.array(col0), np.array(tar0))
        idxs = env1_n + env_a
        col1 = [memory_images.imgs[j][1] // 2 for j in idxs]
        tar1 = [memory_images.imgs[j][1] % 2 for j in idxs]
        corr1 = np.corrcoef(np.array(col1), np.array(tar1))
        print("Color/Label correlations:", "env 0:", corr0[0,1], "env 1:", corr1[0,1])

    train_images = utils.Imagenet_idx(root=data+'/train', transform=None, target_transform=None)
    val_images = utils.Imagenet_idx(root=data+'/val', transform=None, target_transform=None)
    test_images = utils.Imagenet_idx(root=data+'/testgt', transform=None, target_transform=None)

    num_train = len(train_images)
    num_vals = len(val_images)
    num_test = len(test_images)

    col_train, tar_train = [train_images.imgs[j][1] // 2 for j in range(num_train)], [train_images.imgs[j][1] % 2 for j in range(num_train)]
    col_val, tar_val = [val_images.imgs[j][1] // 2 for j in range(num_vals)], [val_images.imgs[j][1] % 2 for j in range(num_vals)]
    col_test, tar_test = [test_images.imgs[j][1] // 2 for j in range(num_test)], [test_images.imgs[j][1] % 2 for j in range(num_test)]
    corr_train = np.corrcoef(np.array(col_train), np.array(tar_train))
    corr_val = np.corrcoef(np.array(col_val), np.array(tar_val))
    corr_test = np.corrcoef(np.array(col_test), np.array(tar_test))
    print("Color/Label correlations:", "train:", corr_train[0,1], "val:", corr_val[0,1], "test:", corr_test[0,1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CMNIST dataset')

    parser.add_argument('--stage1_model', type=str, default='ipirm', help='stage 1 model')
    parser.add_argument('--num_shot', type=str, default='50', help='number of images in each class during stage 1')
    
    args = parser.parse_args()
    
    main(args)
