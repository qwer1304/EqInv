import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import utils
import matplotlib as mpl
import numpy as np
import textwrap

import os
import argparse
from beautifultable import BeautifulTable


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
    num_labels = 2
    num_colors = num_labels
    K = len(env_ref_set[0])
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
    env_idx = list(range(K))

    anchor = 0
    print(f'-----------ANCHOR {anchor}------------')
    # sum of numbers of samples in all envs vs number of "other" samples
    num_samples = sum(len(e) for e in env_ref_set[anchor])
    print(1, f'anchor {anchor}:', 'all envs:', num_samples, \
        'other:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % num_labels != anchor]))

    # number of "other" samples with colors 0/1 in all envs and their sum
    idx = [i for idxs in env_ref_set[anchor] for i in idxs]
    count_c0_o = [memory_images.imgs[i][label] // num_labels for i in idx].count(0)
    count_c1_o = [memory_images.imgs[i][label] // num_labels for i in idx].count(1)

    # number of anchor samples
    indx_a = [j for j in range(len(memory_images)) if memory_images.imgs[j][label] % 2 == anchor]
    # number of anchor samples with colors 0/1 and their sum
    count_c0_a = [memory_images.imgs[j][label] // num_labels for j in indx_a].count(0)
    count_c1_a = [memory_images.imgs[j][label] // num_labels for j in indx_a].count(1)

    # number of "other" samples with colors 0/1 in all envs and their sum
    print(2, f'anchor {anchor}:','R:', count_c0_o, 'G:', count_c1_o, \
        'total other by color:', count_c0_o + count_c1_o, \
        'total other by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % num_labels != anchor]))
    
    # number of anchor samples with colors 0/1 and their sum
    print(3, f'anchor {anchor}:', count_c0_a, count_c1_a, \
        'total anchor by color:', count_c0_a + count_c1_a, \
        'total anchor by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % num_labels == anchor]))
    
    # total number of samples of different colors for anchor
    count_c0_a0 = count_c0_a + count_c0_o
    count_c1_a0 = count_c1_a + count_c1_o
    print(4, 'R:', count_c0_a0, 'G:', count_c1_a0, 'Sum - # samples:', count_c0_a0 + count_c1_a0 - len(memory_images))

    anchor = 1
    print(f'-----------ANCHOR {anchor}------------')
    # sum of numbers of samples in all envs vs number of "other" samples
    num_samples = sum(len(e) for e in env_ref_set[anchor])
    print(5, f'anchor {anchor}:', 'all envs:', num_samples, \
        'other:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % num_labels != anchor]))

    # number of "other" samples with colors 0/1 in all envs and their sum
    idx = [i for idxs in env_ref_set[anchor] for i in idxs]
    count_c0_o = [memory_images.imgs[i][label] // num_labels for i in idx].count(0)
    count_c1_o = [memory_images.imgs[i][label] // num_labels for i in idx].count(1)

    # number of anchor samples
    indx_a = [j for j in range(len(memory_images)) if memory_images.imgs[j][label] % num_labels == anchor]
    # number of anchor samples with colors 0/1 and their sum
    count_c0_a = [memory_images.imgs[j][label] // num_labels for j in indx_a].count(0)
    count_c1_a = [memory_images.imgs[j][label] // num_labels for j in indx_a].count(1)

    # number of "other" samples with colors R/G in all envs and their sum
    print(6, f'anchor {anchor}:','R:', count_c0_o, 'G:', count_c1_o, \
        'total other by color:', count_c0_o + count_c1_o, \
        'total other by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % num_labels != anchor]))
    
    # number of anchor samples with colors R/G and their sum
    print(7, f'anchor {anchor}:', count_c0_a, count_c1_a, \
        'total anchor by color:', count_c0_a + count_c1_a, \
        'total anchor by label:', len([j for j in range(len(memory_images)) if memory_images.imgs[j][label] % num_labels == anchor]))
    
    # total number of samples of different colors for anchor
    count_c0_a1 = count_c0_a + count_c0_o
    count_c1_a1 = count_c1_a + count_c1_o
    print(8, 'R:', count_c0_a0, 'G:', count_c1_a0, 'Sum - # samples:', count_c0_a0 + count_c1_a0 - len(memory_images))
    
    # total number of samples
    print(f'-----------Total number of samples------------')
    print(9, 'Total by color:', int((count_c0_a0 + count_c0_a1 + count_c1_a0 + count_c1_a1) / 2), 'Total # samples:', len(memory_images))
    
    R = 0
    G = 1

    ni = 2 # number of side-by-side plots
    ttl_width = args.title_width
    for k, indeces in env_ref_set.items(): # over anchors, indeces is a tuple of idx tensors
        fig, ax = plt.subplots(1, ni, figsize=(ni*5, 4))
        if ni == 1:
            ax = [ax]
        env_n = [env.tolist() for env in indeces] # lists of "other" samples split between environments
        env_lens = np.array([len(env) for env in env_n])  
        assert np.all(abs(np.diff(env_lens)) <= K-1), f"anchor {k}: number of samples in the environments don't match"
        # Flatten the tuple of lists into one set
        env_n_flat = set().union(*env_n)
        # Compute the set difference
        env_a = list(set(all_idx) - env_n_flat)  # anchor samples
        assert len(env_a) + env_lens.sum() == len(all_idx), f"anchor {k}: number of samples don't add up"

        i = 0
        
        # number of samples for color c in environment e
        env_col = np.zeros((K, num_colors), dtype=int) # (env, col) - environment x color array

        for e in range(env_col.shape[0]):
            env_col[e] = np.array([
                sum([memory_images.imgs[j][label] // num_labels == R for j in env_n[e]]),
                sum([memory_images.imgs[j][label] // num_labels == G for j in env_n[e]])
            ])

        labels = ['R', 'G']
        x = np.arange(len(labels))
        width = 0.35
        offsets = np.linspace(0, K, num=K, endpoint=False)
        colors_hatches = ['red', 'lime']
        colors_env = ['lightsteelblue', 'orange', 'magenta', 'mediumpurple', 'olive']
        xpos = np.zeros((K, len(labels)))
        
        perc = env_col / env_col.sum(axis=0, keepdims=True) * 100 # (env, col)
        
        for e in range(K):
            xpos[e] = x*width*(K+1) + offsets[e]*width
            if hatches_linewidth_supported:
                bar = ax[i].bar(xpos[e], perc[e], width, label=f'env_{e}', hatch="x", color=colors_env[e % len(colors_env)], hatch_linewidth=3.0)
            else:
                bar = ax[i].bar(xpos[e], perc[e], width, label=f'env_{e}', hatch="x", color=colors_env[e % len(colors_env)])

            for j, bc in enumerate(bar):
                bc._hatch_color = mpl.colors.to_rgba(colors_hatches[j])
                bc.stale = True

        ax[i].set_ylabel('Percentage (%)')
        ax[i].set_xlabel('Color')
        ax[i].set_title(textwrap.fill(f'Split of colors R/G between all envs for NON-anchor samples for anchor {k}', width=ttl_width))
        ax[i].set_xticks(np.mean(xpos, axis=0).tolist())
        ax[i].set_xticklabels(labels)
        ax[i].legend(loc='center')
        ax[i].grid(True)
        
        i += 1

        # number of anchor samples for color c 
        col_a = np.array([
            sum([memory_images.imgs[j][label] // num_labels == R for j in env_a]),
            sum([memory_images.imgs[j][label] // num_labels == G for j in env_a])
            ])
            
        perc = (env_col + col_a) / (env_col + col_a).sum(axis=0, keepdims=True) * 100 # (env, col)        
        
        for e in range(K):
            if hatches_linewidth_supported:
                bar = ax[i].bar(xpos[e], perc[e], width, label=f'env_{e}', hatch="x", color=colors_env[e % len(colors_env)], hatch_linewidth=3.0)
            else:
                bar = ax[i].bar(xpos[e], perc[e], width, label=f'env_{e}', hatch="x", color=colors_env[e % len(colors_env)])

            for j, bc in enumerate(bar):
                bc._hatch_color = mpl.colors.to_rgba(colors_hatches[j])
                bc.stale = True

        ax[i].set_ylabel('Percentage (%)')
        ax[i].set_xlabel('Color')
        ax[i].set_title(textwrap.fill(f'Split of colors R/G between all envs for NON-anchor samples for anchor {k}', width=ttl_width))
        ax[i].set_xticks(np.mean(xpos, axis=0).tolist())
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
            
        col_a = np.tile(col_a, (K, 1))  # repeat vertically, env x col
        env_tar = np.array([[np.sum(col_a[j]), np.sum(env_col[j])] for j in range(len(col_a))])
        
        tab = np.hstack((env_col, col_a, env_tar))

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

        corre_na = []
        r=np.random.default_rng()
        fraction = 0.1

        for e in range(K):
            idxs_a = r.choice(len(env_a), int(fraction*len(env_a)), replace=False) 
            idxs = env_n[e] + [env_a[j] for j in idxs_a]
            corre_na.append(col_label_corr(idxs))

        import warnings

        warnings.filterwarnings(
            "ignore",
            message="'BeautifulTable.__iter__' has been deprecated",
            category=FutureWarning
        )

        warnings.filterwarnings(
            "ignore",
            message="'BeautifulTable.__len__' has been deprecated",
            category=FutureWarning
        )
       
        # Setting up the 2nd level table
        table2 = [BeautifulTable() for _ in range(tab.shape[1] // num_colors)] # 2 columns per group - R, G
        for j in range(len(table2)):
            table2[j].columns.header = ["R","G"]
            table2[j].border.left = ''
            table2[j].border.right = ''
            table2[j].border.top = ''
            table2[j].border.bottom = ''
            table2[j].columns.padding_left = 0
            table2[j].columns.padding_right = 0

        for jj in range(len(table2)): # column groups
            for j in range(tab.shape[0]): # rows
                offset = jj*num_colors
                cols = tab[j][offset:offset+num_colors].tolist()
                table2[jj].rows.append(cols)

            # MUST come AFTER rows creation!!!!!!!!!
            table2[jj].rows.header = [f"Env {j}" for j in range(tab.shape[0])]

        table2a = BeautifulTable()
        table2a.columns.header = [" "]
        table2a.border.left = ''
        table2a.border.right = ''
        table2a.border.top = ''
        table2a.border.bottom = ''
        table2a.columns.padding_left = 0
        table2a.columns.padding_right = 0
        for j in range(len(corre_na)):
            table2a.rows.append([corre_na[j]])
        # MUST come AFTER rows creation!!!!!!!!!
        table2a.rows.header = [f"Env {j}" for j in range(tab.shape[0])]
        
        # Setting up the 1st level table
        table1 = BeautifulTable(maxwidth=80)
        table1.columns.header = ['non-anchor', 'anchor', 'label', 'color/label corr']
        table1.rows.append([*table2, table2a])
        table1.border.left = ''
        table1.border.right = ''
        table1.border.top = ''
        table1.border.bottom = ''
        table1.columns.padding_left = 0
        table1.columns.padding_right = 0

        # Setting up the 0th level table
        table0 = BeautifulTable()
        table0.columns.header = [f"Anchor {k}"]
        table0.rows.append([str(table1)])
        table0.columns.padding_left = 0
        table0.columns.padding_right = 0

        print(table0)

        # Optional: align all columns center
        #for i in range(num_cols):
        #    table.columns.alignment[i] = BeautifulTable.ALIGN_CENTER    
                                    
        #print(f"Color/Label correlations {fraction}*pos+neg:", f'anchor: {k}', [f"env {e}: {corre_na[e]:.3f}" for e in range(K)])

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
