# This file is part of GroomRL by S. Carrazza and F. A. Dreyer

from groomrl.Groomer import RSD
from groomrl.JetTree import *
from groomrl.read_data import Jets
from groomrl.tools import mass
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _smooth(arr, window: int):
    """
    Apply a simple rolling-mean smoother.

    Returns
    -------
    smoothed : np.ndarray
        The smoothed values (length ``len(arr) - window + 1``).
    x_vals : np.ndarray
        Episode indices aligned with the *centre* of each window, so the
        smoothed curve sits on top of the raw scatter without phase shift.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window or window <= 1:
        return arr, np.arange(len(arr))
    kernel   = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    half     = window // 2
    x_vals   = np.arange(half, half + len(smoothed))
    return smoothed, x_vals


#----------------------------------------------------------------------
def print_stats(name, data, mass_ref=80.385, output_folder='./', background=False):
    """Print statistics on the mass distribution."""
    r_plain = np.array(data)-mass_ref
    m = np.median(r_plain)
    a = np.mean(r_plain)
    s = np.std(r_plain)
    fn = '%s/diagnostics%s.txt' % (output_folder, '' if not background else '_bkg')
    with open(fn,'a+') as f:
        print('%s:\tmedian-diff %.2f\tavg-diff %.2f\tstd-diff %.2f' % (name, m, a, s),
              file=f)

#----------------------------------------------------------------------
def plot_mass(groomer, sample_fn, mass_ref=80.385, output_folder='./', nev=-1,
              background=False, zcut=0.05, beta=1.0):
    """Plot the mass distribution and output some statistics."""
    reader  = Jets(sample_fn, nev)
    rsd = RSD(zcut=zcut, beta=beta)
    events  = reader.values()
    jets    = []
    groomed_jets = []
    rsd_jets = []
    for jet in events:
        groomed_jet = groomer(jet)
        rsd_jet = rsd(jet)
        jets.append(np.array([jet.px(),jet.py(),jet.pz(),jet.E()]))
        groomed_jets.append(groomed_jet)
        rsd_jets.append(rsd_jet)

    mplain = mass(jets)
    mdqn   = mass(groomed_jets)
    mrsd   = mass(rsd_jets)
    
    bins = np.arange(0, 401, 2)
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(18,14))
    plt.hist(mplain, bins=bins, color='C0', alpha=0.3, label='plain')
    plt.hist(mrsd, bins=bins, alpha=0.4, color='C2',
             label='RSD $(z_\\mathrm{cut}='+'{},\\beta={})$'.format(zcut,beta))
    plt.hist(mdqn,     bins=bins, facecolor='none', edgecolor='C3', lw=2,
             label='DQN-Grooming', hatch="\\")

    plt.xlim((0,200))
    plt.legend()
    fn = '%s/mass%s.pdf' % (output_folder, '' if not background else '_bkg')
    plt.savefig(fn, bbox_inches='tight')
        
    print_stats('plain   ', mplain, mass_ref=mass_ref, output_folder=output_folder, background=background)
    print_stats('mrsd    ', mrsd  , mass_ref=mass_ref, output_folder=output_folder, background=background)
    print_stats('mdqn    ', mdqn  , mass_ref=mass_ref, output_folder=output_folder, background=background)

#----------------------------------------------------------------------
def plot_lund(groomer, sample_fn, zcut=0.05, beta=1.0, output_folder="./",
              nev=-1, background=False):
    """Plot the lund plane."""
    # set up the reader and get array from file
    xval   = [0.0, 7.0]
    yval   = [-3.0, 7.0]
    reader  = Jets(sample_fn, nev)
    rsd = RSD(zcut=zcut, beta=beta)
    lundImg = LundImage(xval, yval)
    events  = reader.values()
    plain_imgs   = []
    groomed_imgs = []
    rsd_imgs     = []
    for jet in events:
        groomed_tree = groomer(jet, returnTree=True)
        rsd_tree = rsd(jet, returnTree=True)
        groomed_imgs.append(lundImg(groomed_tree))
        plain_imgs.append(lundImg(JetTree(jet)))
        rsd_imgs.append(lundImg(rsd_tree))

    avg_plain   = np.average(plain_imgs, axis=0)
    avg_groomed = np.average(groomed_imgs, axis=0)
    avg_rsd     = np.average(rsd_imgs, axis=0)

    # Plot the result
    fn = '%s/lund%s.pdf' % (output_folder, '' if not background else '_bkg')
    with PdfPages(fn) as pdf:

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(12, 9))
        plt.title('Averaged Lund image DQN-Grooming')
        plt.xlabel('$\ln(R / \Delta)$')
        plt.ylabel('$\ln(k_t / \mathrm{GeV})$')
        plt.imshow(avg_groomed.transpose(), origin='lower', aspect='auto',
                   extent=xval+yval, cmap=plt.get_cmap('BuPu'))
        plt.colorbar()
        pdf.savefig()
        plt.close()

        fig=plt.figure(figsize=(12, 9))
        plt.title('Averaged Lund image plain')
        plt.xlabel('$\ln(R / \Delta)$')
        plt.ylabel('$\ln(k_t / \mathrm{GeV})$')
        plt.imshow(avg_plain.transpose(), origin='lower', aspect='auto',
                   extent=xval+yval, cmap=plt.get_cmap('BuPu'))
        plt.colorbar()
        pdf.savefig()
        plt.close()

        fig=plt.figure(figsize=(12, 9))
        plt.title('Averaged Lund image RSD $(z_\\mathrm{cut}='+'{},\\beta={})$'.format(zcut,beta))
        plt.xlabel('$\ln(R / \Delta)$')
        plt.ylabel('$\ln(k_t / \mathrm{GeV})$')
        plt.imshow(avg_rsd.transpose(), origin='lower', aspect='auto',
                   extent=xval+yval, cmap=plt.get_cmap('BuPu'))
        plt.colorbar()
        pdf.savefig()
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  New SB3-era visualisations
# ══════════════════════════════════════════════════════════════════════════════

def plot_decision_boundary(
    model,
    agent_type: str = "dqn",
    lund_dim: int = 2,
    output_folder: str = "./",
    zcut: float = 0.05,
    beta: float = 1.0,
    resolution: int = 100,
    extra_state: np.ndarray = None,
):
    """
    Visualise the trained policy's groom / keep decision across the full
    (lnz, lnDelta) Lund state-space and overlay the equivalent Recursive
    Soft-Drop (RSD) boundary for comparison.

    The plot answers: *where* in Lund space does the agent decide to groom,
    and how does that learned boundary compare to the classical RSD condition
    ``z < z_cut * Δ^β``?

    Parameters
    ----------
    model : SB3AgentGroom or raw SB3 model (DQN / PPO / MDPO)
        If an ``SB3AgentGroom`` wrapper is passed, its inner ``.model``
        attribute is extracted automatically.
    agent_type : str
        Label shown in the plot title.
    lund_dim : int
        Dimensionality of the observation vector.  Only the first two
        components (lnz, lnDelta) are swept; any extra dimensions are held
        fixed at the values in *extra_state*.
    output_folder : str
        Directory in which ``decision_boundary.pdf`` is written.
    zcut, beta : float
        Soft-Drop parameters for the reference boundary line.
    resolution : int
        Number of grid points per axis (``resolution²`` model calls total).
        100 gives a crisp image in ~1 s on CPU.
    extra_state : np.ndarray, optional
        Fixed values for state components 2 … lund_dim−1 (e.g. psi, lnm,
        lnKt for 5-D runs).  Defaults to all-zeros.

    Notes
    -----
    The decision boundary is evaluated with *deterministic* policy inference,
    i.e. the ε-greedy or stochastic components are disabled.  This gives the
    cleanest boundary but may differ slightly from the average behaviour during
    training.

    Output
    ------
    ``{output_folder}/decision_boundary.pdf``
    """
    # Unwrap SB3AgentGroom → raw SB3 model if needed
    inner = getattr(model, "model", model)

    # Axis bounds from LundCoordinates
    lnz_min,     lnz_max     = float(LundCoordinates.low[0]),  float(LundCoordinates.high[0])
    lnDelta_min, lnDelta_max = float(LundCoordinates.low[1]),  float(LundCoordinates.high[1])

    if extra_state is None:
        extra_state = np.zeros(max(0, lund_dim - 2), dtype=np.float32)

    # ── Build observation grid ─────────────────────────────────────────────
    # action_grid[i_lnz, j_lnDelta]  (meshgrid with indexing='ij')
    lnz_vals     = np.linspace(lnz_min,     lnz_max,     resolution)
    lnDelta_vals = np.linspace(lnDelta_min, lnDelta_max, resolution)

    lnz_g, lnDelta_g = np.meshgrid(lnz_vals, lnDelta_vals, indexing="ij")
    flat_states = np.column_stack(
        [lnz_g.ravel(), lnDelta_g.ravel()]
    ).astype(np.float32)

    if lund_dim > 2:
        n_extra = min(lund_dim - 2, len(extra_state))
        pad = np.tile(extra_state[:n_extra], (len(flat_states), 1)).astype(np.float32)
        flat_states = np.concatenate([flat_states, pad], axis=1)

    # ── Evaluate policy ────────────────────────────────────────────────────
    # Try batched prediction first (fast); fall back to per-sample loop.
    try:
        actions, _ = inner.predict(flat_states, deterministic=True)
    except Exception:
        actions = np.array(
            [int(inner.predict(s, deterministic=True)[0]) for s in flat_states],
            dtype=np.int32,
        )

    action_grid = actions.reshape(resolution, resolution).astype(np.float32)
    # action_grid rows = lnz axis (y), cols = lnDelta axis (x)

    # ── RSD reference boundary: lnz = ln(zcut) + beta * lnDelta ───────────
    lnDelta_line = np.linspace(lnDelta_min, lnDelta_max, 500)
    lnz_bnd      = np.log(zcut) + beta * lnDelta_line
    in_range     = (lnz_bnd >= lnz_min) & (lnz_bnd <= lnz_max)

    # Groom fraction across the grid (headline number in title)
    groom_frac = float(action_grid.mean()) * 100.0

    # ── Plot ───────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(9, 7))

    # imshow: rows ↔ axis-0 ↔ lnz (y-axis), cols ↔ axis-1 ↔ lnDelta (x-axis)
    # origin='lower' so lnz increases upward (more physical orientation)
    im = ax.imshow(
        action_grid,
        origin="lower",
        aspect="auto",
        extent=[lnDelta_min, lnDelta_max, lnz_min, lnz_max],
        cmap="RdBu_r",   # red = groom (1), blue = keep (0)
        vmin=0,
        vmax=1,
        alpha=0.85,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Action  (0 = keep,  1 = groom)", fontsize=12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["keep (0)", "groom (1)"])

    # RSD reference boundary
    ax.plot(
        lnDelta_line[in_range],
        lnz_bnd[in_range],
        "k--",
        lw=2.5,
        label=(
            r"RSD boundary  "
            r"($z_{{\rm cut}}={:.3f},\ \beta={:.1f}$)".format(zcut, beta)
        ),
    )

    # Region labels
    ax.text(
        0.97, 0.97, "keep",
        transform=ax.transAxes, ha="right", va="top",
        color="#2166ac", fontsize=13, fontweight="bold",
    )
    ax.text(
        0.03, 0.03, "groom",
        transform=ax.transAxes, ha="left", va="bottom",
        color="#d6604d", fontsize=13, fontweight="bold",
    )

    ax.set_xlabel(r"$\ln \Delta$", fontsize=14)
    ax.set_ylabel(r"$\ln z$",      fontsize=14)
    ax.set_title(
        f"Policy Decision Boundary — {agent_type.upper()}  "
        f"(groom fraction: {groom_frac:.1f}%)",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=11, framealpha=0.85)

    fig.tight_layout()
    fn = f"{output_folder}/decision_boundary.pdf"
    fig.savefig(fn, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Decision boundary → {fn}")


# ──────────────────────────────────────────────────────────────────────────────

def plot_reward_decomposition(
    history: dict,
    output_folder: str = "./",
    smooth_window: int = 50,
    agent_label: str = "Agent",
):
    """
    Plot per-episode training reward curves decomposed into the mass-reward
    and Soft-Drop reward components.

    The function reads from the ``history`` dict produced by
    :class:`groomrl.SB3AgentGroom.RewardTrackingCallback`.  If the decomposed
    arrays (``episode_reward_mass``, ``episode_reward_SD``) are absent the
    function gracefully falls back to plotting only the total reward curve.

    Layout
    ------
    **Panel 1 – total episode reward**
        Raw (faint) and smoothed curves over training episodes.

    **Panel 2 – reward decomposition** *(only when decomposed data available)*
        Mean per-step mass-reward and SD-reward components per episode, with
        the same smoothing applied.  This reveals whether the agent is
        primarily rewarded for hitting the mass target or for executing
        SD-consistent grooming decisions.

    Parameters
    ----------
    history : dict
        Must contain ``"episode_reward"``.  Optionally also
        ``"episode_reward_mass"`` and ``"episode_reward_SD"``.
    output_folder : str
        Directory in which ``reward_decomposition.pdf`` is written.
    smooth_window : int
        Rolling-average window width (in episodes).
    agent_label : str
        Short label shown in panel titles (e.g. ``"DQN"``, ``"MDPO"``).

    Output
    ------
    ``{output_folder}/reward_decomposition.pdf``
    """
    ep_reward = np.asarray(history.get("episode_reward", []), dtype=float)
    ep_mass   = np.asarray(history.get("episode_reward_mass", []), dtype=float)
    ep_sd     = np.asarray(history.get("episode_reward_SD",   []), dtype=float)

    if len(ep_reward) == 0:
        print("[!] plot_reward_decomposition: empty history – nothing to plot.")
        return

    has_decomp = len(ep_mass) > 0 and len(ep_sd) > 0
    episodes   = np.arange(len(ep_reward))

    n_panels = 2 if has_decomp else 1
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(12, 4.5 * n_panels),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    # ── Panel 1: total episode reward ─────────────────────────────────────
    ax = axes[0]
    ax.plot(episodes, ep_reward, alpha=0.20, color="C0", lw=0.8, label="raw")
    if len(ep_reward) >= smooth_window:
        sm, xs = _smooth(ep_reward, smooth_window)
        ax.plot(xs, sm, color="C0", lw=2.0,
                label=f"smoothed  (w = {smooth_window})")
    ax.axhline(float(np.median(ep_reward)), color="C0", lw=1.2, ls=":",
               alpha=0.6, label=f"median = {np.median(ep_reward):.3f}")
    ax.set_ylabel("Episode reward", fontsize=13)
    ax.set_title(f"{agent_label} – Training reward (total)", fontsize=13)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: decomposed reward ────────────────────────────────────────
    if has_decomp:
        ax2 = axes[1]
        # Align lengths (decomposed arrays may be slightly shorter at run-end)
        n   = min(len(episodes), len(ep_mass), len(ep_sd))
        ep_x = episodes[:n]
        ep_m = ep_mass[:n]
        ep_s = ep_sd[:n]

        ax2.plot(ep_x, ep_m, alpha=0.20, color="C1", lw=0.8)
        ax2.plot(ep_x, ep_s, alpha=0.20, color="C2", lw=0.8)

        if n >= smooth_window:
            sm_m, xs_m = _smooth(ep_m, smooth_window)
            sm_s, xs_s = _smooth(ep_s, smooth_window)
            ax2.plot(xs_m, sm_m, color="C1", lw=2.0,
                     label=f"mass reward  (mean={np.mean(ep_m):.3f})")
            ax2.plot(xs_s, sm_s, color="C2", lw=2.0,
                     label=f"Soft-Drop reward  (mean={np.mean(ep_s):.3f})")

            # Stacked area to show relative contributions
            # Only draw where arrays overlap in x
            x_min_common = max(xs_m[0],  xs_s[0])
            x_max_common = min(xs_m[-1], xs_s[-1])
            if x_min_common < x_max_common:
                # Interpolate to a common grid
                x_common = np.arange(x_min_common, x_max_common + 1)
                interp_m = np.interp(x_common, xs_m, sm_m)
                interp_s = np.interp(x_common, xs_s, sm_s)
                ax2.fill_between(x_common, 0, interp_m, color="C1", alpha=0.12)
                ax2.fill_between(x_common, 0, interp_s, color="C2", alpha=0.12)
        else:
            ax2.plot(ep_x, ep_m, color="C1", lw=1.5, label="mass reward")
            ax2.plot(ep_x, ep_s, color="C2", lw=1.5, label="Soft-Drop reward")

        ax2.set_xlabel("Episode", fontsize=13)
        ax2.set_ylabel("Mean step reward", fontsize=13)
        ax2.set_title("Reward decomposition: mass  vs  Soft-Drop component", fontsize=13)
        ax2.legend(loc="upper left", fontsize=11)
        ax2.grid(True, alpha=0.25)

        # Annotate with average SD fraction
        total_mean = float(np.mean(ep_m[:n] + ep_s[:n]))
        if total_mean != 0.0:
            sd_frac = float(np.mean(ep_s[:n])) / total_mean * 100.0
            ax2.text(
                0.97, 0.05,
                f"avg SD fraction: {sd_frac:.1f}%",
                transform=ax2.transAxes, ha="right", va="bottom",
                fontsize=10, color="grey",
            )
    else:
        axes[-1].set_xlabel("Episode", fontsize=13)
        print(
            "[!] plot_reward_decomposition: decomposed reward arrays not found in history.  "
            "Only the total-reward panel is shown.  Make sure you are using the patched "
            "GroomEnvSB3 (which injects reward_mass / reward_SD into info)."
        )

    fig.tight_layout()
    fn = f"{output_folder}/reward_decomposition.pdf"
    fig.savefig(fn, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Reward decomposition → {fn}")
