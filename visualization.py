import matplotlib.pyplot as plt


def show_eigenfaces(model):
    count = min(
        12,
        model["components"],
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(10, 8),
    )

    for i, ax in enumerate(axes.flat):

        if i >= count:
            ax.axis("off")
            continue

        eigenface = model["eigenfaces"][:, i]

        eigenface = eigenface.reshape(
            model["image_size"]
        )

        ax.imshow(
            eigenface,
            cmap="gray",
        )

        ax.set_title(
            f"Eigenface {i + 1}"
        )

        ax.axis("off")

    fig.suptitle(
        "EigenVision - PCA Eigenfaces",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()
