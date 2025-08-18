import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
from tqdm.auto import tqdm
from scipy.interpolate import griddata

def main(args):
    dir = args.data_dir
    filepaths = [os.path.join(dir, f'val_{args.model}_features_dump.pt'), os.path.join(dir, f'test_{args.model}_features_dump.pt')]

    features = []
    labels = []
    labels_raw = []
    domains = []
    logits = []
    # Load
    for i, fp in tqdm(enumerate(filepaths), leave=False, total=len(filepaths)):
        data = torch.load(fp, map_location=torch.device('cpu'))
        if args.variant == 'EqInv':
            features.append(data['features'].numpy())  # convert to numpy
            targets = data['labels'].numpy()
            labels.append(targets)
            targets_raw = data['labels_raw'].numpy()
            labels_raw.append(targets_raw)
            domains.append(i*np.ones_like(targets))
            logits.append(data['logits'].numpy())
        elif args.variant == 'IP-IRM':
            features.append(data['features'].numpy())  # convert to numpy
            targets = data['labels'].numpy()
            labels.append(targets)
            targets_raw = data['labels_raw'].numpy()
            labels_raw.append(targets_raw)
            domains.append(i*np.ones_like(targets))
            logits.append(data['pred_scores'].numpy())
        else:
            raise ValueError(f"Unknown variant {args.variant}")

    # model parameters
    n_classes = data['n_classes']                     # int
        
    features    = np.concatenate(features, axis=0)
    labels      = np.concatenate(labels, axis=0)
    labels_raw  = np.concatenate(labels_raw, axis=0)
    domains     = np.concatenate(domains, axis=0)
    logits      = np.concatenate(logits, axis=0)
    predicts    = np.argmax(logits, axis=1)

    colors = labels_raw % (2*2) // 2
    digits = labels_raw // (2*2)
    with_digits = np.max(digits) > 0

    vars = []
    if not with_digits:
        factors_labs = ['label', 'color']
        factors = [labels, colors]
    else:
        factors_labs = ['label', 'color', 'digits']
        factors = [labels, colors, digits]
    factor_set = set(range(len(factors)))
    mask = domains==0

    def idx_list(d1, d2, debug=False):
        def idx_flist(d):
            dlist = []
            uds = np.unique(d)
            for ud in uds:
                dlist.append(d==ud)
            return dlist
        dl1 = idx_flist(d1)
        dl2 = idx_flist(d2)
        dlist = []
        for l1 in dl1:
            for l2 in dl2:
                dlist.append(l1 & l2)
        return dlist

    for i in range(len(factors)):
        constant_factors = list(factor_set - {i})
        dlist = idx_list(factors[constant_factors[0]], factors[constant_factors[1]])
        feats = [features[dl & mask] for dl in dlist]
        var_np = np.array([np.var(f, axis=0) for f in feats]) # (nl, D)
        weights_np = np.array([np.sum(f, axis=0) for f in feats])
        var_np = np.average(var_np, axis=0, weights=weights_np)
        vars.append(var_np)
           
    k = 10  # top-k values per array
    arrays_list = vars

    # Step 1: top-k indices per array
    topk_indices_list = [np.argsort(arr)[-k:] for arr in arrays_list]

    # Step 2: combined unique indices
    combined_indices = np.unique(np.concatenate(topk_indices_list))

    # Step 3: sort indices
    combined_indices.sort()
    m = len(combined_indices)

    # Step 4: build value matrix (arrays x combined indices)
    values_matrix = np.array([[arr[i] for i in combined_indices] for arr in arrays_list])

    # Step 5: plotting
    n_arrays = len(arrays_list)
    fig, ax = plt.subplots(figsize=(6, 6))

    # normalize colors
    norm = plt.Normalize(values_matrix.min(), values_matrix.max())
    cmap = plt.cm.nipy_spectral
    
    cell_height = 1  # fixed height
    for col in range(n_arrays):
        bottom = 0
        for row in range(m):
            val = values_matrix[col, row]
            ax.bar(col, cell_height, bottom=row*cell_height, color=cmap(norm(val)), width=0.8, edgecolor='black')

    # labels
    ax.set_xticks(range(n_arrays))
    ax.set_xticklabels([f"{factors_labs[i]}" for i in range(n_arrays)])
    ax.set_yticks(range(m))
    ax.set_yticklabels(combined_indices)
    ax.set_ylabel("Index")
    ax.set_title(f"Top-{k} indices combined (m={m})")

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="variance")

    plt.show()

if __name__ == "__main__":
    # create the top-level parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./misc/')
    parser.add_argument('--model', type=str, default="val_best", choices=["val_best", "test_best", "last"])
    parser.add_argument('--variant', type=str, default="EqInv", choices=["EqInv", "IP-IRM"])

    args = parser.parse_args()

    main(args)
