import re
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess  
import argparse

def extract_number(line, keyword="Acc@1"):
    pattern = rf"{keyword}\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    else:
        return None

def main(args):

    fp = os.path.join(args.dir, args.input_fn)

    # Read lines
    lines = []
    epilogue = False
    with open(fp, "r") as f:
        for line in f:
            if "The best Val accuracy" in line:
                epilogue = True
                break
            lines.append(line)

        # Filter lines starting with '*'
        filtered_lines = [line for line in lines if line.lstrip().startswith("*")]

        # Split
        train_lines = filtered_lines[::3]
        val_lines = filtered_lines[1::3]
        test_lines = filtered_lines[2::3]

        # Extract numbers
        train_acc = [extract_number(line) for line in train_lines]
        val_acc = [extract_number(line) for line in val_lines]
        test_acc = [extract_number(line) for line in test_lines]

        # Remove None
        train_acc = [x for x in train_acc if x is not None]
        val_acc = [x for x in val_acc if x is not None]
        test_acc = [x for x in test_acc if x is not None]

        if epilogue:
            # Now process the rest of the file starting from the current line
            best_val_acc = extract_number(line, keyword="accuracy:")

            for line in f:
                if "* Test:" in line:
                    test_best_val_acc = extract_number(line)        

    ibest_val_acc2, best_val_acc2 = np.argmax(np.array(val_acc)), np.max(np.array(val_acc))
    test_best_val_acc2 = test_acc[ibest_val_acc2]   

    # Correlation
    corr = np.corrcoef(np.array(val_acc), np.array(test_acc))
    print(f"val-test acc correlation: {corr[0,1]}")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(2*5, 4))

    ax[0].plot(range(len(train_acc)), train_acc, label='Train', marker='o')
    ax[0].plot(range(len(val_acc)), val_acc, label='Val', marker='+')
    ax[0].plot(range(len(test_acc)), test_acc, label='Test', marker='x')
    ax[0].scatter(ibest_val_acc2, best_val_acc2, marker='x', label='Best Val', s=50, linewidths=3, color='magenta', zorder=2)
    ax[0].scatter(ibest_val_acc2, test_best_val_acc2, marker='x', label='Best Test', s=50, linewidths=3, color='red', zorder=2)

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Acc")
    ax[0].legend()
    ax[0].set_title("Accuracies")
    ax[0].grid(True)

    ax[1].scatter(val_acc, test_acc, marker='.')
    ax[1].scatter(best_val_acc2, test_best_val_acc2, marker='x', label='Best Val', s=50, linewidths=3, color="magenta")

    ax[1].set_xlabel("Val")
    ax[1].set_ylabel("Test")
    ax[1].set_title("Scatter plot of Test vs Val")
    ax[1].grid(True)

    xlim = ax[1].get_xlim()
    ax[1].text(xlim[0]+1*(xlim[1]-xlim[0])/4,30,"val-test acc correlation: {:.3f}".format(corr[0,1]))

    # Linear regression

    x = np.array(val_acc)
    y = np.array(test_acc)
    # Create line for plotting
    x_fit = np.linspace(np.min(x), np.max(x), 100)

    if args.reg_method == 'polynom':
        # Fit line: polynom
        coeffs = np.polyfit(x, y, deg=args.deg)
        p = np.poly1d(coeffs)
        y_fit = p(x_fit)
    elif args.reg_method == 'exponential':
        # Take logs
        log_y = np.log(y)

        # Linear fit
        b, log_a = np.polyfit(x, log_y, 1)
        a = np.exp(log_a)

        # Evaluate exponential curve
        y_fit = a * np.exp(b * x_fit)
    else:
        raise ValueError(f"Unknown regression method {args.reg_method}")

    # Plot
    ax[1].plot(x_fit, y_fit, color='red', label=f"y = f(x)".replace("+", "+ ").replace("-", "- "))
    ax[1].legend()

    fp = os.path.join(args.dir, args.output_fn)
    plt.savefig(fp, format='jpg')
    subprocess.run(['start', '', os.path.abspath(fp)], shell=True)
    #os.startfile(os.path.abspath(fp))

if __name__ == "__main__":
    # create the top-level parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-fn', type=str, default='20250713_1649_run.txt', help='input file name')
    parser.add_argument('--output-fn', type=str, default='val_test_acc.jpg', help='output file name')
    parser.add_argument('--dir', type=str, default='./misc', help='input file dir')
    parser.add_argument('--reg_method', type=str, choices=['polynom', 'exponential'], default='exponential', help='regression method')
    parser.add_argument('--deg', type=int, default=1, help='polynomial regression degree')

    args = parser.parse_args()

    main(args)
