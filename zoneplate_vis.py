"""
Fully commented Zone-Plate visualization for the thesis.

Purpose
-------
The script creates a Dubois-style quadratic-phase zone plate and passes the
same synthetic input directly through isolated anti-aliasing/downsampling
operators. It is meant for the controlled characterization in Chapter 3,
not as an end-to-end ResNet evaluation.

Representative methods
----------------------
- Unfiltered 2x decimation
- BlurPool-3
- WaveCNet-bior3.3
- PASA-3, group 8
- DABPool-3
- ASAPsp

Literature basis
----------------
Eric Dubois, "Multidimensional Signal and Color Image Processing Using
Lattices", Wiley, 2019, Sec. 2.2.4 ("Zone Plate").

The zone-plate definition used here is

    z(x, y) = cos(pi * (x^2 + y^2) / r0^2)

with local spatial-frequency components

    u(x, y) = x / r0^2
    v(x, y) = y / r0^2

The value 0.30 cycles/original-pixel used at the horizontal/vertical image
boundary is NOT prescribed by Dubois. It is the experimental scaling chosen
for this thesis so that the probe contains frequencies below and above the
0.25 cycles/original-pixel Nyquist limit after 2x downsampling.

Important interpretation notes
------------------------------
- "Unfiltered 2x" is a sampling reference, not the trained ResNet baseline.
- The zone plate is passed directly to the isolated AA/downsampling operator.
  Learned Conv/BN/ReLU operations around the operator are intentionally left
  out.
- PASA and DABPool use learned parameters from a selected trained checkpoint.
- PASA is input-adaptive. Therefore its result is an empirical response to
  this probe, not a classical LTI frequency response.
- The raw Dubois cosine ranges from -1 to +1. In the real ResNet, PASA may
  normally receive post-ReLU activations. This synthetic probe is therefore
  deliberately controlled rather than activation-distribution realistic.
"""

# os is used to construct paths relative to the location of this script.
import os

# NumPy is used to create the coordinate grid and evaluate the zone-plate formula.
import numpy as np

# Matplotlib is used to display and save the generated comparison figure.
import matplotlib.pyplot as plt

# PyTorch is used because the anti-aliasing/downsampling operators are torch modules.
import torch

# aa_models is the project module containing the modified ResNet-18 implementations.
import aa_models


# CONFIGURATION
# __file__ gives the path of this Python file; abspath turns it into an absolute path.
SCRIPT_PATH = os.path.abspath(__file__)

# dirname removes the file name and leaves the directory containing this script.
BASE_DIR = os.path.dirname(SCRIPT_PATH)

# Generated plots are written into a folder called "plots" next to this script.
OUT_DIR = os.path.join(BASE_DIR, "plots")

# Trained model outputs/checkpoints are assumed to live below a folder called "out".
OUT_ROOT = os.path.join(BASE_DIR, "out")

# GPU selects which CUDA device is used if CUDA is available.
GPU = 0

# SEED selects the trained PASA/DABPool checkpoint used for the isolated response.
SEED = 1

# SIZE is the width and height of the synthetic zone plate in pixels.
SIZE = 256

# STAGE=2 means that the isolated operator is extracted from layer2[0] of ResNet-18.
STAGE = 2

# EDGE_FREQUENCY is the thesis-specific local frequency reached at an axis-aligned image edge.
EDGE_FREQUENCY = 0.30

# Construct the expected file path of the trained PASA-3, group-8 checkpoint.
PASA3_CKPT = os.path.join(
    # Start in the common output directory.
    OUT_ROOT,
    # Select the PASA run corresponding to the configured seed and parameters.
    f"resnet18_pasa_seed{SEED}_filter3_group8",
    # Select the best saved checkpoint from that run.
    f"resnet18_pasa_seed{SEED}_filter3_group8_best.pth.tar",
)

# Construct the expected file path of the trained DABPool-3 checkpoint.
DAB3_CKPT = os.path.join(
    # Start in the common output directory.
    OUT_ROOT,
    # Select the DABPool run corresponding to the configured seed and filter size.
    f"resnet18_dab_seed{SEED}_filter3",
    # Select the best saved checkpoint from that run.
    f"resnet18_dab_seed{SEED}_filter3_best.pth.tar",
)


# ZONE-PLATE GENERATION
# Define a function that creates one quadratic-phase zone plate and useful frequency maps.
def generate_zone_plate(size=SIZE, edge_frequency=EDGE_FREQUENCY):
    """
    Generate the controlled Dubois-style zone plate used in the thesis.

    For unit-spaced pixels, the coordinates are centered around zero. For an
    even size such as 256, the axis edge is located at R=127.5.

    The target edge frequency determines r0 through

        u(R, 0) = R / r0^2 = edge_frequency,

    so

        r0^2 = R / edge_frequency.
    """

    # R is the distance from the center to an axis-aligned image boundary in original-pixel units.
    R = (size - 1) / 2.0

    # arange creates the discrete sample positions 0, 1, ..., size-1.
    raw_coords = np.arange(size, dtype=np.float32)

    # Subtract R so that the coordinate origin lies at the image center.
    coords = raw_coords - R

    # meshgrid expands the 1-D coordinate vector into 2-D x and y coordinate arrays.
    xx, yy = np.meshgrid(coords, coords, indexing="xy")

    # Solve R / r0^2 = edge_frequency for r0^2.
    r0_sq = R / edge_frequency

    # Take the positive square root to obtain the scale parameter r0 itself.
    r0 = np.sqrt(r0_sq)

    # Evaluate the quadratic phase phi(x,y)=pi*(x^2+y^2)/r0^2.
    phase = np.pi * (xx**2 + yy**2) / r0_sq

    # Apply cosine to the phase to obtain the real Dubois-style zone plate in [-1,+1].
    plate = np.cos(phase)

    # The local horizontal frequency follows from (1/(2*pi))*d(phi)/dx = x/r0^2.
    local_u = xx / r0_sq

    # The local vertical frequency follows from (1/(2*pi))*d(phi)/dy = y/r0^2.
    local_v = yy / r0_sq

    # The radial magnitude is useful for diagnostics, although Nyquist is applied component-wise here.
    local_radial = np.sqrt(local_u**2 + local_v**2)

    # Return float32 arrays because PyTorch model tensors are normally float32.
    return (
        # Return the actual zone-plate image.
        plate.astype(np.float32),
        # Return the local horizontal spatial-frequency map.
        local_u.astype(np.float32),
        # Return the local vertical spatial-frequency map.
        local_v.astype(np.float32),
        # Return the local radial frequency magnitude map.
        local_radial.astype(np.float32),
        # Return r0^2 as a normal Python float for readable logging.
        float(r0_sq),
        # Return r0 as a normal Python float for readable logging.
        float(r0),
    )


# MODEL LOADING
# Define a helper that constructs one modified ResNet-18 and optionally restores a checkpoint.
def load_model(
    # aa_type selects the anti-aliasing method in aa_models.
    aa_type,
    # filter_size selects the spatial support where that parameter is relevant.
    filter_size=3,
    # wavelet_type selects the wavelet family for WaveCNet/DWT.
    wavelet_type="haar",
    # pasa_group selects the number of PASA channel groups.
    pasa_group=8,
    # checkpoint is None for methods that do not need learned AA parameters here.
    checkpoint=None,
    # device is the CPU or CUDA device on which the model will run.
    device="cpu",
):
    # Construct the appropriate modified ResNet-18 using the project's model factory.
    model = aa_models.resnet18(
        # Pass the anti-aliasing method name.
        aa_type=aa_type,
        # Pass the selected filter size.
        filter_size=filter_size,
        # Pass the selected wavelet family.
        wavelet_type=wavelet_type,
        # Pass the PASA grouping parameter.
        pasa_group=pasa_group,
        # Tiny ImageNet has 200 classes, matching the trained checkpoints.
        num_classes=200,
    )

    # Only enter the checkpoint-loading block when a checkpoint path was supplied.
    if checkpoint is not None:
        # Check that the configured checkpoint actually exists on disk.
        if not os.path.isfile(checkpoint):
            # Stop instead of silently using an untrained adaptive operator.
            raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint}")

        # Read the checkpoint onto CPU first so it is independent of the GPU used during training.
        weights = torch.load(checkpoint, map_location="cpu")

        # Some training scripts save a dictionary containing "state_dict"; others save it directly.
        state_dict = weights.get("state_dict", weights)

        # Copy the saved parameter values into the newly constructed model.
        model.load_state_dict(state_dict)

    # Switch the model into inference mode so BatchNorm/dropout behave deterministically.
    model.eval()

    # Move all model parameters and buffers to the selected device.
    model.to(device)

    # Return the model so the isolated AA operator can be extracted from it.
    return model


# EXTRACT THE ISOLATED DOWNSAMPLING OPERATOR
# Define a helper that returns the AA operator and the channel count it expects.
def get_operator(model, aa_type):
    """
    Extract the isolated downsampling operator from layer2[0].

    Current project integration:

        BlurPool / PASA: Conv -> BN -> ReLU -> AA
        WaveCNet / DABPool: Conv -> AA -> BN -> ReLU
        ASAPsp: AA -> Conv

    The learned convolution, BatchNorm and ReLU are deliberately excluded from
    this experiment because the goal is to characterize the downsampling
    mechanism itself.
    """

    # getattr retrieves layer2 when STAGE is 2 without hard-coding the attribute name twice.
    stage = getattr(model, f"layer{STAGE}")

    # The first BasicBlock of a new stage is the block that performs spatial downsampling.
    block = stage[0]

    # In this implementation BlurPool and PASA are attached after the first ReLU.
    if aa_type in ("blur", "pasa"):
        # Return the AA module and the number of channels produced by conv1.
        return block.aa_after_relu1, block.conv1.out_channels

    # In this implementation WaveCNet/DWT and DABPool are attached after conv1.
    if aa_type in ("dwt", "dab"):
        # Return that AA module and the number of conv1 output channels it expects.
        return block.aa_after_conv1, block.conv1.out_channels

    # In this implementation ASAPsp is placed before conv1.
    if aa_type == "asap":
        # Return the pre-conv AA module and therefore the number of conv1 input channels.
        return block.aa_before_conv1, block.conv1.in_channels

    # Reaching this line means that the method name is not handled by this extraction code.
    raise ValueError(f"Unknown aa_type: {aa_type}")


# RUN THE CONTROLLED PROBE
# no_grad disables gradient tracking because this experiment performs inference only.
@torch.no_grad()
# Define a helper that applies one isolated anti-aliasing/downsampling operator.
def run_operator(model, aa_type, plate, device):
    # Find the operator inside the model and determine how many channels it expects.
    operator, channels = get_operator(model, aa_type)

    # Convert the HxW NumPy zone plate into a PyTorch tensor on the selected device.
    x = torch.from_numpy(plate).to(device)

    # Add a batch axis and a channel axis, changing HxW into 1x1xHxW.
    x = x[None, None]

    # Repeat the same spatial zone plate across every expected feature channel.
    x = x.repeat(1, channels, 1, 1)

    # Pass the tensor through the isolated AA/downsampling operator only.
    out = operator(x)

    # Average across channels so the response can be shown as a single 2-D image.
    out = out.mean(dim=1)

    # Remove the single batch dimension, leaving an HxW response image.
    out = out[0]

    # Move the result to CPU and convert it to NumPy for Matplotlib.
    return out.cpu().numpy()


# no_grad also disables gradient tracking for the unfiltered reference path.
@torch.no_grad()
# Define a factor-2 decimation reference with no low-pass filtering at all.
def run_unfiltered_reference(plate, device):
    # Convert the HxW NumPy input into a PyTorch tensor on the chosen device.
    x = torch.from_numpy(plate).to(device)

    # Add one batch and one channel dimension, obtaining shape 1x1xHxW.
    x = x[None, None]

    # ::2 keeps every second row and every second column, which is direct 2x decimation.
    y = x[:, :, ::2, ::2]

    # Remove the batch and channel dimensions, leaving a 2-D image.
    y = y[0, 0]

    # Move the reference result back to CPU and convert it to NumPy.
    return y.cpu().numpy()


# MAIN
# Define the main function so this file can also be imported without immediately running the experiment.
def main():
    # Select the configured GPU if CUDA is available; otherwise fall back to CPU.
    device = torch.device(
        # This is the CUDA device string, e.g. "cuda:0".
        f"cuda:{GPU}"
        # Check whether PyTorch can use CUDA on this machine.
        if torch.cuda.is_available()
        # Use CPU when CUDA is unavailable.
        else "cpu"
    )

    # Print the selected device to make the run log reproducible.
    print(f"Device: {device}")

    # Generate the zone plate, local-frequency maps and the derived scale parameter.
    (
        # The actual zone-plate image.
        zone_plate,
        # Local horizontal spatial-frequency map u(x,y).
        local_u,
        # Local vertical spatial-frequency map v(x,y).
        local_v,
        # Local radial frequency magnitude map.
        local_radial,
        # Derived squared scale parameter r0^2.
        r0_sq,
        # Derived scale parameter r0.
        r0,
    ) = generate_zone_plate(
        # Use the configured image size of 256.
        size=SIZE,
        # Use the thesis-specific axis-edge frequency of 0.30 cycles/original-pixel.
        edge_frequency=EDGE_FREQUENCY,
    )

    # Recompute the axis radius only for readable diagnostic output below.
    R = (SIZE - 1) / 2.0

    # With one sample per original pixel, the original Nyquist limit is 0.5 cycles/pixel.
    original_nyquist = 0.5

    # After factor-2 decimation, Nyquist is 0.25 cycles/original-pixel in original coordinates.
    downsampled_nyquist_original_units = 0.25

    # In a corner both u and v reach 0.30, so the radial magnitude is sqrt(2)*0.30.
    corner_radial_frequency = np.sqrt(2.0) * EDGE_FREQUENCY

    # Print the generated image dimensions.
    print(f"Zone-plate size: {SIZE} x {SIZE}")

    # Print R because it is used directly to derive r0^2.
    print(f"Axis radius R: {R:.4f} original pixels")

    # Print the deliberately chosen axis-edge frequency.
    print(f"Target axis-edge frequency: {EDGE_FREQUENCY:.4f} cycles/original-pixel")

    # Print r0^2 so the derivation R/0.30 can be checked numerically.
    print(f"r0^2: {r0_sq:.4f}")

    # Print r0 itself because this is the scale symbol used in the zone-plate formula.
    print(f"r0: {r0:.4f}")

    # Print the Nyquist limit of the original pixel grid.
    print(f"Original Nyquist: {original_nyquist:.4f} cycles/original-pixel")

    # Print the Nyquist limit after factor-2 downsampling in the same original-pixel units.
    print(
        # Prefix explaining the reported value.
        "Nyquist after 2x downsampling: "
        # Insert the numeric value with four decimal places.
        f"{downsampled_nyquist_original_units:.4f} cycles/original-pixel"
    )

    # Print the radial corner magnitude as a diagnostic, not as the square-grid Nyquist limit.
    print(
        # Prefix explaining what this diagnostic quantity is.
        "Corner radial frequency magnitude: "
        # Insert the computed sqrt(2)*0.30 value.
        f"{corner_radial_frequency:.4f} cycles/original-pixel"
    )

    # Explicitly remind the reader that the 2-D sampling limit is component-wise in u and v.
    print("Note: 2-D Nyquist is applied component-wise to u and v, not as a radial cutoff.")

    # Store all representative method configurations in one list for a uniform evaluation loop.
    configs = [
        # First representative configuration: BlurPool with 3x3 support.
        (
            # Human-readable label for plots and logs.
            "BlurPool-3",
            # Internal aa_models identifier.
            "blur",
            # Spatial filter support.
            3,
            # Placeholder wavelet value; unused by BlurPool.
            "haar",
            # Placeholder PASA grouping value; unused by BlurPool.
            8,
            # Fixed BlurPool requires no learned checkpoint for this isolated operator.
            None,
        ),
        # Second representative configuration: WaveCNet using bior3.3.
        (
            # Human-readable label for plots and logs.
            "WaveCNet-bior3.3",
            # Internal aa_models identifier for DWT-based downsampling.
            "dwt",
            # Placeholder filter size expected by the common model factory.
            3,
            # Wavelet family used for the representative WaveCNet response.
            "bior3.3",
            # Placeholder PASA grouping value; unused by WaveCNet.
            8,
            # The fixed DWT operator does not require a learned AA checkpoint here.
            None,
        ),
        # Third representative configuration: PASA with 3x3 support and eight groups.
        (
            # Human-readable label for plots and logs.
            "PASA-3 (g=8)",
            # Internal aa_models identifier.
            "pasa",
            # PASA spatial neighborhood size.
            3,
            # Placeholder wavelet value; unused by PASA.
            "haar",
            # Number of PASA channel groups.
            8,
            # Restore the trained PASA parameters selected above.
            PASA3_CKPT,
        ),
        # Fourth representative configuration: DABPool with 3x3 support.
        (
            # Human-readable label for plots and logs.
            "DABPool-3",
            # Internal aa_models identifier.
            "dab",
            # DABPool spatial support.
            3,
            # Placeholder wavelet value; unused by DABPool.
            "haar",
            # Placeholder PASA grouping value; unused by DABPool.
            8,
            # Restore the trained DABPool parameters selected above.
            DAB3_CKPT,
        ),
        # Fifth representative configuration: ASAPsp.
        (
            # Human-readable label for plots and logs.
            "ASAPsp",
            # Internal aa_models identifier.
            "asap",
            # Placeholder filter size expected by the common model factory.
            3,
            # Placeholder wavelet value; unused by ASAPsp.
            "haar",
            # Placeholder PASA grouping value; unused by ASAPsp.
            8,
            # This isolated ASAPsp operator does not require a learned AA checkpoint here.
            None,
        ),
    ]

    # Start the output dictionary with the direct-decimation reference.
    outputs = {
        # The dictionary key is the human-readable panel label.
        "Unfiltered 2x": run_unfiltered_reference(
            # Give the reference the exact same zone plate used for every AA method.
            zone_plate,
            # Perform the tensor slicing on the selected device.
            device,
        )
    }

    # Iterate once over every representative anti-aliasing configuration.
    for (
        # label is used in progress logs and the final figure.
        label,
        # aa_type tells aa_models which architecture variant to construct.
        aa_type,
        # filter_size selects the spatial support where relevant.
        filter_size,
        # wavelet_type selects the DWT basis where relevant.
        wavelet_type,
        # pasa_group sets PASA's channel grouping where relevant.
        pasa_group,
        # checkpoint supplies learned parameters where required.
        checkpoint,
    ) in configs:
        # Print the current method so a long-running script shows its progress.
        print(f"Running {label}...")

        # Construct the appropriate model and restore learned AA parameters if needed.
        model = load_model(
            # Select the current anti-aliasing method.
            aa_type=aa_type,
            # Select its configured spatial support.
            filter_size=filter_size,
            # Select its configured wavelet family.
            wavelet_type=wavelet_type,
            # Select its configured PASA grouping.
            pasa_group=pasa_group,
            # Supply the configured checkpoint or None.
            checkpoint=checkpoint,
            # Place the model on the same device as the probe tensor.
            device=device,
        )

        # Apply only the isolated downsampling operator to the common zone plate.
        output = run_operator(
            # Supply the model that contains the operator.
            model=model,
            # Supply the type so get_operator knows where the module is stored.
            aa_type=aa_type,
            # Supply the common synthetic input.
            plate=zone_plate,
            # Supply the computation device.
            device=device,
        )

        # Store the returned response under its human-readable method name.
        outputs[label] = output

        # Print shape and extrema as a simple sanity check for each method response.
        print(
            # Show the downsampled spatial shape.
            f"  shape={output.shape}, "
            # Show the smallest response value.
            f"min={output.min():.4f}, "
            # Show the largest response value.
            f"max={output.max():.4f}"
        )

    # Combine every displayed value so all panels can use one common intensity range.
    all_values = np.concatenate(
        # Include the full-resolution input itself.
        [zone_plate.ravel()]
        # Add every downsampled response after flattening it to one dimension.
        + [image.ravel() for image in outputs.values()]
    )

    # Choose the largest absolute value, but never use less than 1 because the input is in [-1,+1].
    max_abs = max(1.0, float(np.max(np.abs(all_values))))

    # Use the negative common maximum as the lower grayscale limit.
    vmin = -max_abs

    # Use the positive common maximum as the upper grayscale limit.
    vmax = max_abs

    # Define the exact panel order for the main qualitative comparison figure.
    panels = [
        # Original synthetic probe before any downsampling.
        ("Input Zone Plate", zone_plate),
        # Direct factor-2 decimation with no anti-aliasing filter.
        ("Unfiltered 2x", outputs["Unfiltered 2x"]),
        # Fixed spatial low-pass filtering followed by downsampling.
        ("BlurPool-3", outputs["BlurPool-3"]),
        # Wavelet-based downsampling with bior3.3.
        ("WaveCNet-bior3.3", outputs["WaveCNet-bior3.3"]),
        # Input-adaptive spatial downsampling with PASA.
        ("PASA-3 (g=8)", outputs["PASA-3 (g=8)"]),
        # Learned/adaptive blur downsampling with DABPool.
        ("DABPool-3", outputs["DABPool-3"]),
        # Spectral/frequency-selective downsampling with ASAPsp.
        ("ASAPsp", outputs["ASAPsp"]),
    ]

    # Create eight subplot cells so seven method panels fit into a compact 2x4 layout.
    fig, axes = plt.subplots(
        # Create two rows.
        2,
        # Create four columns.
        4,
        # Set the physical figure size in inches.
        figsize=(12, 6),
    )

    # Flatten the 2-D subplot array so the panels can be filled in a simple loop.
    axes = axes.ravel()

    # Iterate over one subplot axis and one panel at a time.
    for ax, (title, image) in zip(axes, panels):
        # Draw the current 2-D response image.
        ax.imshow(
            # Supply the actual 2-D array.
            image,
            # Use grayscale because pixel intensity, not color, carries the information.
            cmap="gray",
            # Apply the same lower intensity limit to every panel.
            vmin=vmin,
            # Apply the same upper intensity limit to every panel.
            vmax=vmax,
            # Use nearest-neighbor only for display so Matplotlib does not visually smooth the data.
            interpolation="nearest",
        )

        # Add the method name above the current subplot.
        ax.set_title(
            # Use the current human-readable panel title.
            title,
            # Keep the title compact enough for the multi-panel layout.
            fontsize=10,
        )

        # Hide tick labels and axes because they do not add useful information to this image comparison.
        ax.axis("off")

    # Iterate over any unused subplot cells after all seven panels have been placed.
    for ax in axes[len(panels):]:
        # Hide the unused eighth subplot completely.
        ax.axis("off")

    # Add one title describing the purpose of the whole figure.
    fig.suptitle(
        # Call it a qualitative zone-plate response, not a classical transfer function.
        "Qualitative Zone-Plate Response of Representative Downsampling Methods",
        # Use a slightly larger font than the individual panel titles.
        fontsize=13,
    )

    # Let Matplotlib adjust spacing to reduce overlaps between titles and panels.
    plt.tight_layout()

    # Create the plots output directory if it does not exist yet.
    os.makedirs(
        # Create the directory configured above.
        OUT_DIR,
        # Do not fail when the directory already exists.
        exist_ok=True,
    )

    # Construct the output file name for the PNG version.
    png_path = os.path.join(
        # Place the file in the plots directory.
        OUT_DIR,
        # Use a descriptive name that identifies the Dubois-based visualization.
        "zone_plate_representative_methods_dubois.png",
    )

    # Construct the output file name for the PDF version.
    pdf_path = os.path.join(
        # Place the file in the same plots directory.
        OUT_DIR,
        # Use the same base name for the PDF version.
        "zone_plate_representative_methods_dubois.pdf",
    )

    # Save a lossless high-resolution PNG suitable for checking the result and thesis drafts.
    plt.savefig(
        # Save to the path constructed above.
        png_path,
        # Use 300 dpi for normal print-quality raster output.
        dpi=300,
        # Remove unnecessary whitespace around the final figure.
        bbox_inches="tight",
    )

    # Save a PDF version, which is convenient for LaTeX inclusion.
    plt.savefig(
        # Save to the PDF path constructed above.
        pdf_path,
        # Remove unnecessary whitespace here as well.
        bbox_inches="tight",
    )

    # Close the figure so repeated executions do not keep figures in memory.
    plt.close()

    # Print a short completion message.
    print("\nSaved:")

    # Print the exact location of the PNG file.
    print(f"  {png_path}")

    # Print the exact location of the PDF file.
    print(f"  {pdf_path}")


# This guard is true only when the script is executed directly rather than imported as a module.
if __name__ == "__main__":
    # Call main to run the complete controlled zone-plate experiment.
    main()
