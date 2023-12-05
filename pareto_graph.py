import matplotlib.pyplot as plt
import numpy as np

def plot_scatter_graph(data, title, x_label, y_label, x_scale="linear", y_scale="linear"):
    """
    data: list of tuples (label, x, y)
    """
    fig, ax = plt.subplots()
    for label, x, y, compression in data:
        label_text = "blue" if compression else "orange"
        ax.scatter(x, y, c=label_text)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha="left")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    plt.scatter([], [], color='blue', label='Compressed', marker='o')
    plt.scatter([], [], color='orange', label='Uncompressed', marker='o')
    plt.legend(scatterpoints=1, labelspacing=1, loc='upper right')
    #ax.legend()
    plt.savefig(f"{title}.png", bbox_inches="tight")
    

if __name__ == "__main__":
    # example usage
    # First number is bpb and second number is MAUVE and third is 1 if a compression algorithm
    data = [
        ("Byte", 5.32, 1, 0),
        ("BPE", 4.24, 2, 0),
        ("WordPiece", 3, 3, 0),
        ("RLE", 2, 4, 1),
        ("BWT + RLE", 1, 5, 1),
        ("LZ77", 0, 6, 1),
        ("DEFLATE", 15.29, 7, 1),
    ]
    plot_scatter_graph(data, "Next Byte Prediction", "Bits per Byte", "MAUVE")

