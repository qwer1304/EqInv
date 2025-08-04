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
    if not args.skip_embedding:
        print('Importing ... ', end="")
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        from umap import UMAP
        print('Done!')

        filepaths = [os.path.join(dir, f'val_{args.model}_features_dump.pt'), os.path.join(dir, f'test_{args.model}_features_dump.pt')]
            
        features = []
        labels = []
        labels_raw = []
        domains = []
        logits = []
        # Load
        for i, fp in tqdm(enumerate(filepaths), leave=False, total=len(filepaths)):
            data = torch.load(fp, map_location=torch.device('cpu'))
            features.append(data['features'].numpy())  # convert to numpy
            targets = data['labels'].numpy()
            labels.append(targets)
            targets_raw = data['labels_raw'].numpy()
            labels_raw.append(targets_raw)
            print(len(targets),len(targets_raw))
            domains.append(i*np.ones_like(targets))
            logits.append(data['logits'].numpy())
        features    = np.concatenate(features, axis=0)
        labels      = np.concatenate(labels, axis=0)
        labels_raw  = np.concatenate(labels_raw, axis=0)
        domains     = np.concatenate(domains, axis=0)
        logits      = np.concatenate(logits, axis=0)
        predicts    = np.argmax(logits, axis=1)
        
        # model parameters
        weights   = data['head_weights'].detach().numpy() # (num_classes, embed_dim)
        biases    = data['head_bias'].detach().numpy()    # (num_classes,)
        n_classes = data['n_classes']                     # int
        
        if args.method == 'tsne':
            tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=0, verbose=2, max_iter=args.max_iter, init='pca')
            features_2d = tsne.fit_transform(np.concatenate([features, weights], axis=0))

        elif args.method == 'pca':
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(np.concatenate([features, weights], axis=0))
       
        elif args.method == 'umap':
            umap = UMAP(n_components=2, n_neighbors=args.n_neighbors, min_dist=args.min_dist, verbose=True)
            features_2d = umap.fit_transform(np.concatenate([features, weights], axis=0))
           
        else:
            raise ValueError(f"Unknown method {args.method}")
            
        weights_2d = features_2d[-n_classes:,:]  
        features_2d = features_2d[:-n_classes,:]

        print('Saving embeddings ... ', end="")
        np.save(os.path.join(dir,f"features_2d_{args.method}_{args.model}.npy"), features_2d)
        np.save(os.path.join(dir,f"weights_2d_{args.model}.npy"), weights_2d)
        np.save(os.path.join(dir,f"labels_{args.model}.npy"), labels)
        np.save(os.path.join(dir,f"labels_raw_{args.model}.npy"), labels_raw)
        np.save(os.path.join(dir,f"domains_{args.model}.npy"), domains)
        np.save(os.path.join(dir,f"biases_{args.model}.npy"), biases)
        np.save(os.path.join(dir,f"logits_{args.model}.npy"), logits)
        np.save(os.path.join(dir,f"predicts_{args.model}.npy"), predicts)
        print('Done!')

    else:
        print('Loading embeddings ... ', end="")
        features_2d = np.load(os.path.join(dir,f"features_2d_{args.method}_{args.model}.npy"))
        weights_2d = np.load(os.path.join(dir,f"weights_2d_{args.model}.npy"))
        labels = np.load(os.path.join(dir,f"labels_{args.model}.npy"))
        labels_raw = np.load(os.path.join(dir,f"labels_raw_{args.model}.npy"))
        domains = np.load(os.path.join(dir,f"domains_{args.model}.npy"))
        biases = np.load(os.path.join(dir,f"biases_{args.model}.npy"))
        logits = np.load(os.path.join(dir,f"logits_{args.model}.npy"))
        predicts = np.load(os.path.join(dir,f"predicts_{args.model}.npy"))
        n_classes = weights_2d.shape[0]
        print('Done!')

    fig, axs = plt.subplots(2, 2, figsize=(2*7, 2*5))  # 2 row, 2 columns
    axs = axs.flatten()
    cmap = plt.cm.tab10
    n_cols = 10
    coloffset = 0
    #cmap = plt.cm.Set1
    #n_cols = 9
    #coloffset = 0

    i = 0
    lls = labels
    target = "class"
    print('Plotting classes ... ', end="")

    u_lls = np.unique(lls)

    handles = []
    markers = ['8', '*']

    for j, l in enumerate(u_lls):
        fidx = (lls == l)
        domain_feat = cmap((j + coloffset) % n_cols)
        domain_w = cmap((j + len(u_lls)) % n_cols)
        marker_w = markers[j % len(markers)]
        size_w = max(300-j*100,50)

        # Features scatter
        axs[i].scatter(features_2d[fidx][:, 0], features_2d[fidx][:, 1], alpha=0.2, s=4+4*j, marker=".", color=domain_feat, zorder=len(u_lls)-j)

        # Weights scatter
        axs[i].scatter(weights_2d[j, 0], weights_2d[j, 1], alpha=1.0, s=size_w, marker=marker_w, color=domain_w, zorder=len(u_lls)+1)

        # Create proxy handles for legend
        feature_proxy = mlines.Line2D([], [], color=domain_feat, marker="o", linestyle="None",
                                      markersize=6, label=f"feature {target}: {l}")
        weight_proxy = mlines.Line2D([], [], color=domain_w, marker=marker_w, linestyle="None",
                                     markersize=14, label=f"weight {target}: {l}")

        handles.append(feature_proxy)
        handles.append(weight_proxy)

    # Fix axis limits after plotting to avoid autoscaling
    original_xlim = axs[i].get_xlim()
    original_ylim = axs[i].get_ylim()

    # 3. Compute the effective 2D classifier normal
    w0_2d = weights_2d[0]
    w1_2d = weights_2d[1]
    w_diff_2d = w0_2d - w1_2d  # shape [1, 2]
    w_diff_2d = w_diff_2d.flatten()  # shape [2]
    
    # 4. Project the 1D bias (difference)
    b_diff = biases[0] - biases[1]

    # 6. Compute midpoint of decision boundary
    # We'll use the average of projected weights as a point on the line
    midpoint = (w0_2d + w1_2d).flatten() / 2
    
    # 7. Plot the decision boundary
    # The boundary is orthogonal to w_diff_2d and passes through the midpoint
    # So we define a line: (x - x0, y - y0) \dot w_diff_2d = 0
    
    x_vals = np.linspace(features_2d[:, 0].min(), features_2d[:, 0].max(), 200)
    # solve for y using (x - x0) * dx + (y - y0) * dy = 0 => y = y0 - dx/dy * (x - x0)
    dx, dy = w_diff_2d
    if abs(dy) < 1e-6:
        # Vertical line
        dec_bound = axs[i].axvline(x=midpoint[0], color='black', linestyle='--')
    else:
        y_vals = midpoint[1] - (dx / dy) * (x_vals - midpoint[0])
        dec_bound = axs[i].plot(x_vals, y_vals, 'k--')
    decision_boundary_proxy = mlines.Line2D([], [], color='black', linestyle='--', label='decision boundary')
    handles.insert(2,decision_boundary_proxy)

    axs[i].set_xlim(original_xlim)
    axs[i].set_ylim(original_ylim)
    
    
    # Add annotated arrows to '0' and '1' decision spaces. 
    normal = np.array([dx, dy*1.0], dtype=float)
    normal /= np.linalg.norm(normal)
    
    # Tangent vector (perpendicular to normal)
    tangent = np.array([-dy, dx], dtype=float)
    tangent /= np.linalg.norm(tangent)
    
    arrow_length = 5.0
    tangent_offset = 0.5  # how far to slide along the boundary

    for direction, label, color, align, t_offset in [
        (+1, f"Decision: {u_lls[0]}", cmap((0 + len(u_lls)) % n_cols), 'left',  +1.1*tangent_offset),
        (-1, f"Decision: {u_lls[1]}", cmap((1 + len(u_lls)) % n_cols), 'right', -9.0*tangent_offset)]:

        # Slide along the boundary by t_offset
        base = np.array(midpoint) + tangent * t_offset
        
        # Tail and head of the arrow
        tail = base
        head = base + direction * normal * arrow_length

        # Arrow: from tail to head
        axs[i].annotate("",
                        xy=head,
                        xytext=tail,
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))

        # Label: offset from arrowhead in the same direction
        label_pos = head + direction * normal * 4.0

        axs[i].text(*label_pos, label,
                    fontsize=12, color=color,
                    ha=align, va='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))
    
    print('Done!')
  
    axs[i].legend(handles=handles, loc='upper center', ncol=len(u_lls))
    axs[i].set_title(f"{args.method} domained by {target}")

    i += 1
    lls = domains
    target = "domain"
    print('Plotting domains ... ', end="")
    
    u_lls = np.unique(lls)

    handles = []

    for j, l in enumerate(u_lls):
        fidx = (lls == l)
        domain_feat = cmap((j + coloffset) % n_cols)
        # Features scatter
        axs[i].scatter(features_2d[fidx][:, 0], features_2d[fidx][:, 1], alpha=0.2, s=4+4*j, marker=".", color=domain_feat, zorder=len(u_lls)-j)
        # Create proxy handles for legend
        feature_proxy = mlines.Line2D([], [], color=domain_feat, marker="o", linestyle="None",
                                      markersize=6, label=f"{target}: {l}")
        handles.append(feature_proxy)
   
    print('Done')

    axs[i].legend(handles=handles, loc='upper center', ncol=len(u_lls))
    axs[i].set_title(f"{args.method} domained by {target}")
    
    i += 1
    lls = predicts != labels
    test_samples = domains == 1
    target = "predicts"
    coloffset = 2
    print('Plotting predicts ... ', end="")
    
    u_lls = np.unique(lls)

    handles = []

    for j, l in enumerate(u_lls):
        fidx = (lls == l) & test_samples
        domain_feat = cmap((coloffset + j) % n_cols)
        # Features scatter
        axs[i].scatter(features_2d[fidx][:, 0], features_2d[fidx][:, 1], alpha=0.2, s=4+4*j, marker=".", color=domain_feat, zorder=len(u_lls)-j)
        # Create proxy handles for legend
        feature_proxy = mlines.Line2D([], [], color=domain_feat, marker="o", linestyle="None",
                                      markersize=6, label=f"{target}: {'Wrong' if l else 'Correct'}")
        handles.append(feature_proxy)
   
    print('Done')

    axs[i].legend(handles=handles, loc='upper center', ncol=len(u_lls))
    axs[i].set_title(f"{args.method} domained by {target} @ test")

    i += 1
    lls = labels_raw // n_classes
    target = "colors"
    coloffset = 2
    print('Plotting colors ... ', end="")
    
    u_lls = np.unique(lls)
    
    handles = []

    for j, l in enumerate(u_lls):
        fidx = (lls == l)
        domain_feat = cmap((coloffset + j) % n_cols)
        # Features scatter
        axs[i].scatter(features_2d[fidx][:, 0], features_2d[fidx][:, 1], alpha=0.2, s=4+4*j, marker=".", color=domain_feat, zorder=len(u_lls)-j)
        # Create proxy handles for legend
        feature_proxy = mlines.Line2D([], [], color=domain_feat, marker="o", linestyle="None",
                                      markersize=6, label=f"{target}: {'Red' if l else 'Green'}")
        handles.append(feature_proxy)
   
    print('Done')

    axs[i].legend(handles=handles, loc='upper center', ncol=len(u_lls))
    axs[i].set_title(f"{args.method} domained by {target}")

    fig.suptitle(f"model: {args.model}")

    plt.savefig(os.path.join(dir, f"{args.method}_{args.model}.jpg"), format='jpg')
    os.startfile(os.path.abspath(os.path.join(dir, f"{args.method}_{args.model}.jpg")))

    
if __name__ == "__main__":
    # create the top-level parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./misc/')
    parser.add_argument('--skip_embedding', action='store_true', help='skip embedding; load instead')
    parser.add_argument('--model', type=str, default="val_best", choices=["val_best", "test_best", "last"])
    subparsers = parser.add_subparsers(dest='method', help='embedding method: tsne, pca, umap')

    # create the parser for the "tsne" command
    parser_tsne = subparsers.add_parser('tsne', help='tsne')
    parser_tsne.add_argument('--max_iter', type=int, default=1000, help='at least 250')
    parser_tsne.add_argument('--perplexity', type=int, default=30)

    # create the parser for the "pca" command
    parser_pca = subparsers.add_parser('pca', help='pca')

    # create the parser for the "umap" command
    parser_umap = subparsers.add_parser('umap', help='umap')
    parser_umap.add_argument('--n_neighbors', type=int, default=15)
    parser_umap.add_argument('--min_dist', type=float, default=0.1, help="[0.0,1.0]")
       
    args = parser.parse_args()

    main(args)
