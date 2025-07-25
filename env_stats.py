import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import utils
import matplotlib as mpl
import numpy as np
import textwrap

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

    fp = args.fp
    env_ref_set = torch.load(fp)
    data = args.data
    memory_images = utils.Imagenet_idx(root=data + '/train', transform=None, target_transform=None)
    num_samples = len(memory_images)
    all_idx = list(range(num_samples))

    """
    env_ref_set is a dictionary over class labels.
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
    print(1, f'anchor {anchor}:', 'env0 + env1:', len(env_ref_set[anchor][env0]) + len(env_ref_set[anchor][env1]), \
        'other:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 != anchor]))

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
    print(2, f'anchor {anchor}:',f'env {env0}:', count_c0_o, f'env {env1}:', count_c1_o, \
        'total other by color:', count_c0_o + count_c1_o, \
        'total other by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 != anchor]))
    
    # number of anchor samples with colors 0/1 and their sum
    print(3, f'anchor {anchor}:', count_c0_a, count_c1_a, \
        'total anchor by color:', count_c0_a + count_c1_a, \
        'total anchor by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 == anchor]))
    
    # total number of samples of different colors for anchor
    count_c0_a0 = count_c0_a + count_c0_o
    count_c1_a0 = count_c1_a + count_c1_o
    print(4, 'Color 0:', count_c0_a0, 'Color 1:', count_c1_a0, 'Sum - # samples:', count_c0_a0 + count_c1_a0 - len(memory_images))

    anchor = 1
    print(f'-----------ANCHOR {anchor}------------')
    # sum of numbers of samples in env0 + env1 vs number of "other" samples
    print(5, f'anchor {anchor}:', 'env0 + env1:', len(env_ref_set[anchor][env0]) + len(env_ref_set[anchor][env1]), \
        'other:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 != anchor]))

    # number of samples with colors 0 in env0 and env1
    count_c0_o = [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env0]].count(0) + \
                 [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env1]].count(0)
    # number of samples with colors 1 in env0 and env1
    count_c1_o = [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env0]].count(1) + \
                 [memory_images.imgs[j][label] // 2 for j in env_ref_set[anchor][env1]].count(1)

    # number of anchor samples
    indx_a = [j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 == anchor]
    # number of anchor samples with colors 0
    count_c0_a = [memory_images.imgs[j][label] // 2 for j in indx_a].count(0)
    # number of anchor samples with colors 1
    count_c1_a = [memory_images.imgs[j][label] // 2 for j in indx_a].count(1)

    # number of samples with colors 0/1 in env0 and env1 and their sum
    print(6, f'anchor {anchor}:',f'env {env0}:', count_c0_o, f'env {env1}:', count_c1_o, \
        'total other by color:', count_c0_o + count_c1_o, \
        'total other by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 != anchor]))

    # number of anchor samples with colors 0/1 and their sum
    print(7, f'anchor {anchor}:', count_c0_a, count_c1_a, \
        'total anchor by color:', count_c0_a + count_c1_a, \
        'total anchor by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 == anchor]))
    
    # total number of samples of different colors for anchor
    count_c0_a1 = count_c0_a + count_c0_o
    count_c1_a1 = count_c1_a + count_c1_o
    print(8, 'Color 0:', count_c0_a1, 'Color 1:', count_c1_a1, 'Sum - # samples:', count_c0_a1 + count_c1_a1 - len(memory_images))
    
    # total number of samples
    print(f'-----------Total number of samples------------')
    print(9, 'Total by color:', (count_c0_a0 + count_c0_a1 + count_c1_a0 + count_c1_a1) / 2, 'Total # samples:', len(memory_images))
    
    R = 0
    G = 1

    ni = 2
    ttl_width = args.title_width
    for k, indeces in env_ref_set.items(): # over anchors, indeces is a tuple
        fig, ax = plt.subplots(1, ni, figsize=(ni*5, 4))
        if ni == 1:
            ax = [ax]
        env_n = [indeces[0].tolist(), indeces[1].tolist()] # "other" samples split between environments
        assert abs(len(env_n[0]) - len(env_n[1])) <= 1, f"anchor {k}: number of samples in the two environments don't match"
        env_a = list(set(all_idx) - set(env_n[0]) - set(env_n[1])) # anchor samples
        assert len(env_a) + len(env_n[0]) + len(env_n[1]) == len(all_idx), f"anchor {k}: number of samples don't add up"

        i = 0
        # number of samples for color c in environment e
        env_col = np.zeros((len(env_ref_set), 2), dtype=int) # (env, col) - environment x color array

        for e in range(env_col.shape[0]):
            env_col[e] = np.array([
                sum([memory_images.imgs[j][label] // 2 == R for j in env_n[e]]),
                sum([memory_images.imgs[j][label] // 2 == G for j in env_n[e]])
            ])

        perc = env_col / env_col.sum(axis=0, keepdims=True) * 100 # (env, col)
        
        labels = ['R', 'G']
        x = np.arange(len(labels))
        width = 0.35
        colors_hatches = ['red', 'lime']
        if hatches_linewidth_supported:
            bar = ax[i].bar(x - width/2, perc[0], width, label='env_0', hatch="x", color='lightsteelblue', hatch_linewidth=3.0)
        else:
            bar = ax[i].bar(x - width/2, perc[0], width, label='env_0', hatch="x", color='lightsteelblue')

        for j, bc in enumerate(bar):
            bc._hatch_color = mpl.colors.to_rgba(colors_hatches[j])
            bc.stale = True

        if hatches_linewidth_supported:
            bar = ax[i].bar(x + width/2, perc[1], width, label='env_1', hatch="x", color='orange', hatch_linewidth=3.)
        else:
            bar = ax[i].bar(x + width/2, perc[1], width, label='env_1', hatch="x", color='orange')

        for j, bc in enumerate(bar):
            bc._hatch_color = mpl.colors.to_rgba(colors_hatches[j])
            bc.stale = True

        ax[i].set_ylabel('Percentage (%)')
        ax[i].set_xlabel('Color')
        ax[i].set_title(textwrap.fill(f'Split of colors R/G between envs 0/1 for NON-anchor samples for anchor {k}', width=ttl_width))
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels)
        ax[i].legend(loc='center')
        ax[i].grid(True)
        
        i += 1

        # number of anchor samples for color c 
        col_a = np.array([
            sum([memory_images.imgs[j][label] // 2 == R for j in env_a]),
            sum([memory_images.imgs[j][label] // 2 == G for j in env_a])
            ])

        perc = (env_col + col_a) / (env_col + col_a).sum(axis=0, keepdims=True) * 100 # (env, col)
        
        labels = ['R', 'G']
        x = np.arange(len(labels))
        width = 0.35
        colors_hatches = ['red', 'lime']
        if hatches_linewidth_supported:
            bar = ax[i].bar(x - width/2, perc[0], width, label='env_0', hatch="x", color='lightsteelblue', hatch_linewidth=3.0)
        else:
            bar = ax[i].bar(x - width/2, perc[0], width, label='env_0', hatch="x", color='lightsteelblue')

        for j, bc in enumerate(bar):
            bc._hatch_color = mpl.colors.to_rgba(colors_hatches[j])
            bc.stale = True

        if hatches_linewidth_supported:
            bar = ax[i].bar(x + width/2, perc[1], width, label='env_1', hatch="x", color='orange', hatch_linewidth=3.)
        else:
            bar = ax[i].bar(x + width/2, perc[1], width, label='env_1', hatch="x", color='orange')

        for j, bc in enumerate(bar):
            bc._hatch_color = mpl.colors.to_rgba(colors_hatches[j])
            bc.stale = True

        ax[i].set_ylabel('Percentage (%)')
        ax[i].set_xlabel('Color')
        ax[i].set_title(textwrap.fill(f'Split of colors R/G between envs 0/1 for ALL samples for anchor {k}', width=ttl_width))
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels)
        ax[i].legend(loc='center')
        ax[i].grid(True)
        
        plt.tight_layout()

        # How to handle final display or saving
        if is_notebook():
            # In notebook, it automatically displays inline
            pass
        elif is_headless():
            plt.savefig(f"plot_{k}.png")
            plt.close()  # Close figure to avoid duplicate output and free memory
            print(f"Plot saved to plot{k}.png (headless mode).")
        else:
            plt.show(block = (k == len(env_ref_set)-1 and i == 1))
            
        print(f"Anchor {k}:")
        print("non-anchor: env vs color")
        print(env_col)
        print("anchor: env vs color")
        col_a = np.vstack([col_a, col_a])
        print(col_a)
        
        print("env vs label:")
        env_tar = np.array([[np.sum(col_a[0]), np.sum(env_col[0])],  
                            [np.sum(col_a[1]), np.sum(env_col[1])]])
        print(env_tar)
                            
        def col_label_corr(idxs):
            col = [memory_images.imgs[j][label] // 2 for j in idxs]
            tar = [memory_images.imgs[j][label] % 2 for j in idxs]
            x = np.asarray(col)
            y = np.asarray(tar)
            std_x = np.std(x)
            std_y = np.std(y)
    
            if std_x == 0 or std_y == 0:
                return 0.0  # or np.nan, depending on what you want
            else:
                return np.corrcoef(x, y)[0, 1]

        idxs = env_n[0] + env_a
        corr0_na = col_label_corr(idxs)
        
        idxs = env_n[1] + env_a
        corr1_na = col_label_corr(idxs)

        print("Color/Label correlations pos+neg:", f'achor: {k}', "env 0:", corr0_na, "env 1:", corr1_na)

    train_images = utils.Imagenet_idx(root=data+'/train', transform=None, target_transform=None)
    val_images = utils.Imagenet_idx(root=data+'/val', transform=None, target_transform=None)
    test_images = utils.Imagenet_idx(root=data+'/testgt', transform=None, target_transform=None)

    num_train = len(train_images)
    num_vals = len(val_images)
    num_test = len(test_images)

    col_train, tar_train = [train_images.imgs[j][label] // 2 for j in range(num_train)], [train_images.imgs[j][label] % 2 for j in range(num_train)]
    col_val, tar_val = [val_images.imgs[j][label] // 2 for j in range(num_vals)], [val_images.imgs[j][label] % 2 for j in range(num_vals)]
    col_test, tar_test = [test_images.imgs[j][label] // 2 for j in range(num_test)], [test_images.imgs[j][label] % 2 for j in range(num_test)]
    corr_train = np.corrcoef(np.array(col_train), np.array(tar_train))
    corr_val = np.corrcoef(np.array(col_val), np.array(tar_val))
    corr_test = np.corrcoef(np.array(col_test), np.array(tar_test))
    print("Color/Label correlations:", "train:", corr_train[0,1], "val:", corr_val[0,1], "test:", corr_test[0,1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CMNIST dataset')

    parser.add_argument('--fp', type=str, help='path of cluster file')
    parser.add_argument('--data', type=str, default='./data/DataSets/CMNIST_wcolor/64')
    parser.add_argument('--title_width', type=int, default=40)
    
    args = parser.parse_args()
    
    main(args)
