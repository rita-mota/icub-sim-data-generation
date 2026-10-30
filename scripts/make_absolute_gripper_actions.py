"""Create absolute binary gripper commands from recorded hand closure.

The collector's seventh EE value is an achieved closure measurement. It is not
an action: contact can stop a grasp long before the normalized value reaches
1.0. This script detects sustained closing/opening motion, converts each event
to a latched command, and aligns the event with the action immediately before
the first observed movement.

Command convention:
    1.0 = open / keep open
    0.0 = close / keep holding

The source arrays are never modified. With ``--write``, each episode receives:

* ``gripper/gripper_actions_absolute.npy``: shape (N - 1,)
* ``ee_deltas_not_absolute_gripper.npy``: an untouched backup of the original
  ``ee_deltas.npy``
* ``ee_deltas.npy``: the first six corrected EE deltas plus the binary gripper
  command, if the original array has shape (N - 1, 7+)
* ``gripper/absolute_gripper_conversion.json``: parameters and detected transitions
* ``gripper/absolute_gripper_conversion.png``: optional diagnostic plot

Run without ``--write`` first to inspect proposed transitions. Use
``--manual-transitions`` to override automatic detection for difficult cases.

Examples:
    python scripts/make_absolute_gripper_actions.py recordings/t2
    python scripts/make_absolute_gripper_actions.py recordings/t2 --write --plot
    # Migrate episodes previously processed by the old output convention:
    python scripts/make_absolute_gripper_actions.py recordings/t2 --write --plot --overwrite
    python scripts/make_absolute_gripper_actions.py path/to/episode_001 \
        --manual-transitions close:180,open:932 --write --plot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np


OPEN_COMMAND = 1.0
CLOSE_COMMAND = 0.0


@dataclass(frozen=True)
class Transition:
    action_index: int
    command: Literal["open", "close"]
    closure_before: float | None
    closure_after: float | None


def is_episode_dir(path: Path) -> bool:
    return (path / "ee_poses.npy").is_file()


def find_episode_dirs(input_path: Path) -> list[Path]:
    if is_episode_dir(input_path):
        return [input_path]
    return sorted({path.parent for path in input_path.rglob("ee_poses.npy")})


def median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Suppress short sensor spikes with a centered median filter.

    Each output sample is the median of ``window`` neighbouring measurements.
    Unlike a mean, the median is barely affected by a single unusually high or
    low value, which makes the later motion test less likely to invent a gripper
    transition.  At the ends, repeating the nearest sample keeps the output the
    same length as the input.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd number")
    if window == 1:
        return values.astype(np.float64, copy=True)

    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1)


def infer_initial_command(
    smoothed: np.ndarray,
    requested: Literal["auto", "open", "closed"],
    direction: float,
) -> float:
    """Infer whether the episode starts open (1) or closed (0).

    After correcting the signal direction, small values mean "more open" and
    large values mean "more closed".  We estimate the two ends of the observed
    range with the 5th and 95th percentiles and split that range at its midpoint:

        midpoint = (5th percentile + 95th percentile) / 2

    The median of the first ten samples is classified as open when it lies below
    that midpoint, and closed otherwise.  Medians and percentiles are used so a
    few noisy measurements do not determine the initial binary state.
    """
    if requested == "open":
        return OPEN_COMMAND
    if requested == "closed":
        return CLOSE_COMMAND

    # Put both sensor conventions into one coordinate system:
    #   canonical closure c[t] = direction * measured closure[t]
    # Thus an increase in c always represents closing, irrespective of whether
    # the raw sensor itself increases or decreases when the fingers close.
    canonical = direction * smoothed
    low = float(np.percentile(canonical, 5.0))
    high = float(np.percentile(canonical, 95.0))
    initial = float(np.median(canonical[: min(10, len(canonical))]))
    return OPEN_COMMAND if initial <= (low + high) / 2.0 else CLOSE_COMMAND


def _first_significant_step(
    one_step_change: np.ndarray,
    start: int,
    stop: int,
    sign: float,
    step_threshold: float,
) -> int:
    """Find the first one-frame movement that agrees with a detected event."""
    stop = min(stop, len(one_step_change))
    for index in range(max(0, start), stop):
        if sign * one_step_change[index] >= step_threshold:
            return index
    return max(0, start)


def detect_transitions(
    smoothed: np.ndarray,
    initial_command: float,
    direction: float,
    motion_window: int,
    confirmation: int,
    threshold: float,
    min_gap: int,
) -> list[Transition]:
    """Detect alternating close/open events and backdate them to motion onset.

    Let c[t] be the direction-corrected closure measurement.  For action index
    t, the script measures motion over at most ``motion_window`` frames:

        D[t] = c[t + 1] - c[max(0, t - motion_window + 1)]

    If the current command is open, the next possible event is closing, so the
    required sign is s=+1.  If it is closed, the next event must be opening, so
    s=-1.  A transition is accepted only when

        s * D[t + k] >= threshold,  for every k in [0, confirmation - 1].

    Requiring several consecutive windows rejects isolated noise spikes.  Once
    accepted, the command flips and remains latched until the opposite motion is
    detected: open (1) -> closed (0), or closed (0) -> open (1).
    """
    if motion_window < 1:
        raise ValueError("--motion-window must be at least 1")
    if confirmation < 1:
        raise ValueError("--confirmation must be at least 1")
    if min_gap < 0:
        raise ValueError("--min-gap cannot be negative")

    # Convert the signal so positive motion always means closing.
    canonical = direction * smoothed

    # one_step[t] = c[t + 1] - c[t] is the motion associated with action t:
    # observation t --action t--> observation t+1.
    one_step = np.diff(canonical)

    # cumulative[t] is D[t] from the docstring.  Summing one-step differences
    # would give the same result because the intermediate terms telescope:
    # (c[a+1]-c[a]) + ... + (c[t+1]-c[t]) = c[t+1]-c[a].
    cumulative = np.empty_like(one_step)
    for action_index in range(len(one_step)):
        first_observation = max(0, action_index - motion_window + 1)
        cumulative[action_index] = canonical[action_index + 1] - canonical[first_observation]

    transitions: list[Transition] = []
    current_command = initial_command
    action_index = 0
    # The windowed test tells us that an event happened, but not exactly which
    # action began it.  Treat half the average required per-frame motion as a
    # meaningful onset step, with 1e-6 preventing a zero tolerance.
    step_threshold = max(1e-6, threshold / (2.0 * motion_window))

    while action_index + confirmation <= len(cumulative):
        # Commands must alternate.  While open (1), look only for positive
        # closing motion; while closed (0), look only for negative opening motion.
        sign = 1.0 if current_command == OPEN_COMMAND else -1.0
        confirmed = np.all(
            sign * cumulative[action_index : action_index + confirmation] >= threshold
        )
        if not confirmed:
            action_index += 1
            continue

        # Backdate the transition from the confirming window to the first
        # significant one-frame step.  This aligns the new binary command with
        # the action that caused the first visible change, not with the later
        # frame at which enough evidence accumulated.
        onset = _first_significant_step(
            one_step,
            start=action_index - motion_window + 1,
            stop=action_index + confirmation,
            sign=sign,
            step_threshold=step_threshold,
        )
        if transitions and onset - transitions[-1].action_index < min_gap:
            action_index += 1
            continue

        next_command = CLOSE_COMMAND if current_command == OPEN_COMMAND else OPEN_COMMAND
        transitions.append(
            Transition(
                action_index=int(onset),
                command="close" if next_command == CLOSE_COMMAND else "open",
                closure_before=float(smoothed[onset]),
                closure_after=float(smoothed[onset + 1]),
            )
        )
        current_command = next_command
        action_index = max(action_index + 1, onset + min_gap)

    return transitions


def parse_manual_transitions(specification: str, num_actions: int) -> list[Transition]:
    transitions: list[Transition] = []
    previous_index = -1
    previous_command: str | None = None

    for item in specification.split(","):
        try:
            command, raw_index = item.strip().lower().split(":", maxsplit=1)
            action_index = int(raw_index)
        except ValueError as exc:
            raise ValueError(
                "Manual transitions must look like 'close:180,open:932'"
            ) from exc

        if command not in {"open", "close"}:
            raise ValueError(f"Unknown manual command: {command!r}")
        if not 0 <= action_index < num_actions:
            raise ValueError(
                f"Manual action index {action_index} is outside [0, {num_actions - 1}]"
            )
        if action_index <= previous_index:
            raise ValueError("Manual transition indices must be strictly increasing")
        if command == previous_command:
            raise ValueError("Consecutive manual transitions must alternate commands")

        transitions.append(
            Transition(
                action_index=action_index,
                command=command,
                closure_before=None,
                closure_after=None,
            )
        )
        previous_index = action_index
        previous_command = command

    return transitions


def build_commands(
    num_actions: int,
    initial_command: float,
    transitions: list[Transition],
) -> np.ndarray:
    """Turn sparse transitions into a binary, piecewise-constant action signal.

    Mathematically, if transition i occurs at action index tau_i with requested
    value q_i in {0, 1}, then the output is q[t] = q_i for
    tau_i <= t < tau_(i+1).  Before the first transition it equals the inferred
    initial value.  Assigning each value to ``[action_index:]`` implements this
    latch: the gripper keeps its last command until another event changes it.
    """
    commands = np.full(num_actions, initial_command, dtype=np.float32)
    current_command = initial_command

    for transition in transitions:
        requested = OPEN_COMMAND if transition.command == "open" else CLOSE_COMMAND
        if requested == current_command:
            raise ValueError(
                f"Transition {transition.command}:{transition.action_index} does not change "
                "the current command. Check --initial-command or the transition list."
            )
        commands[transition.action_index :] = requested
        current_command = requested

    return commands


def save_plot(
    output_file: Path,
    raw: np.ndarray,
    smoothed: np.ndarray,
    commands: np.ndarray,
    transitions: list[Transition],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plot requires matplotlib") from exc

    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(raw, color="0.75", linewidth=1.0, label="measured closure")
    axis.plot(smoothed, color="tab:blue", linewidth=1.5, label="smoothed closure")
    axis.step(
        np.arange(len(commands)),
        commands,
        where="post",
        color="tab:orange",
        linewidth=1.5,
        label="absolute command (1=open, 0=close)",
    )
    for transition in transitions:
        axis.axvline(transition.action_index, color="tab:red", alpha=0.5, linewidth=1.0)
        axis.text(
            transition.action_index,
            1.02,
            transition.command,
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
        )
    axis.set_xlabel("Observation/action index")
    axis.set_ylabel("Value")
    axis.set_title(output_file.parent.name)
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def convert_episode(episode_dir: Path, args: argparse.Namespace) -> None:
    poses = np.load(episode_dir / "ee_poses.npy")
    if poses.ndim != 2 or poses.shape[1] <= args.gripper_column:
        raise ValueError(
            f"{episode_dir}: expected gripper column {args.gripper_column} in "
            f"ee_poses.npy, got shape {poses.shape}"
        )
    if len(poses) < 2:
        raise ValueError(f"{episode_dir}: at least two observations are required")

    raw = poses[:, args.gripper_column].astype(np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{episode_dir}: gripper signal contains non-finite values")

    smoothed = median_smooth(raw, args.smooth_window)
    direction = 1.0 if args.closure_direction == "increasing" else -1.0
    initial_command = infer_initial_command(smoothed, args.initial_command, direction)

    # Robust observed range R = P95(c) - P5(c).  The automatic motion threshold
    # is max(minimum_threshold, threshold_fraction * R): it scales with the
    # episode's actual gripper motion but can never become too sensitive when R
    # is tiny.  A user-supplied --motion-threshold replaces this calculation.
    dynamic_range = float(
        np.percentile(direction * smoothed, 95.0)
        - np.percentile(direction * smoothed, 5.0)
    )
    threshold = (
        args.motion_threshold
        if args.motion_threshold is not None
        else max(args.minimum_threshold, args.threshold_fraction * dynamic_range)
    )

    if args.manual_transitions:
        transitions = parse_manual_transitions(args.manual_transitions, len(poses) - 1)
    else:
        transitions = detect_transitions(
            smoothed=smoothed,
            initial_command=initial_command,
            direction=direction,
            motion_window=args.motion_window,
            confirmation=args.confirmation,
            threshold=threshold,
            min_gap=args.min_gap,
        )

    commands = build_commands(len(poses) - 1, initial_command, transitions)
    transition_text = ", ".join(
        f"{event.command}:{event.action_index}" for event in transitions
    ) or "none"
    initial_text = "open" if initial_command == OPEN_COMMAND else "closed"
    print(
        f"{episode_dir}: initial={initial_text}, range={dynamic_range:.6f}, "
        f"threshold={threshold:.6f}, transitions=[{transition_text}]"
    )

    if not args.write:
        return

    gripper_dir = episode_dir / args.output_dir_name
    command_file = gripper_dir / args.output_name
    combined_file = episode_dir / args.combined_output_name
    backup_delta_file = episode_dir / args.backup_delta_name
    legacy_combined_file = episode_dir / args.legacy_combined_name
    metadata_file = gripper_dir / "absolute_gripper_conversion.json"
    plot_file = gripper_dir / "absolute_gripper_conversion.png"
    outputs = [command_file, metadata_file]
    if args.plot:
        outputs.append(plot_file)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{episode_dir}: output already exists ({existing[0].name}); use --overwrite"
        )

    gripper_dir.mkdir(parents=True, exist_ok=True)
    np.save(command_file, commands)
    combined_written = False
    combined_warning: str | None = None
    legacy_combined_removed = False
    if not args.no_combined:
        # On the first conversion, preserve the original action array before
        # replacing the canonical file. On later --overwrite runs, always use
        # that backup as the source so it can never be contaminated by a
        # previously converted binary gripper column.
        if backup_delta_file.is_file():
            source_delta_file = backup_delta_file
        elif combined_file.is_file():
            source_delta_file = combined_file
            if legacy_combined_file.is_file():
                print(
                    f"  Detected legacy {legacy_combined_file.name}; "
                    f"preserving {combined_file.name} as the original backup."
                )
        elif legacy_combined_file.is_file():
            source_delta_file = legacy_combined_file
            combined_warning = (
                f"{combined_file.name} was missing; restored it from legacy "
                f"{legacy_combined_file.name}, but no original continuous-"
                "gripper action array was available to back up"
            )
        else:
            source_delta_file = combined_file

        if not source_delta_file.is_file():
            combined_warning = (
                f"neither {combined_file.name} nor "
                f"{backup_delta_file.name} nor "
                f"{legacy_combined_file.name} exists"
            )
        else:
            deltas = np.load(source_delta_file)
            if deltas.ndim != 2 or deltas.shape[0] != len(commands) or deltas.shape[1] < 7:
                combined_warning = (
                    f"{source_delta_file.name} has shape {deltas.shape}; expected "
                    f"({len(commands)}, 7+) after next-action conversion"
                )
            else:
                if (
                    not backup_delta_file.exists()
                    and source_delta_file != legacy_combined_file
                ):
                    np.save(backup_delta_file, deltas)
                combined = deltas[:, :7].copy()
                combined[:, 6] = commands
                np.save(combined_file, combined)
                combined_written = True

                # The old combined output contains no information that is not
                # now present in the canonical file. Remove it only after the
                # replacement has been written successfully.
                if (
                    legacy_combined_file.is_file()
                    and legacy_combined_file != combined_file
                    and legacy_combined_file != backup_delta_file
                ):
                    legacy_combined_file.unlink()
                    legacy_combined_removed = True
                    print(
                        f"  Removed redundant legacy file "
                        f"{legacy_combined_file.name}."
                    )

    metadata = {
        "source_episode": str(episode_dir),
        "num_observations": int(len(poses)),
        "num_actions": int(len(commands)),
        "command_semantics": {"1": "open", "0": "close_or_hold"},
        "source_signal": f"ee_poses.npy column {args.gripper_column}",
        "initial_command": initial_text,
        "closure_direction": args.closure_direction,
        "smooth_window": args.smooth_window,
        "motion_window": args.motion_window,
        "confirmation": args.confirmation,
        "minimum_transition_gap": args.min_gap,
        "observed_dynamic_range": dynamic_range,
        "motion_threshold": threshold,
        "automatic_detection": not bool(args.manual_transitions),
        "transitions": [asdict(event) for event in transitions],
        "command_file": str(command_file.relative_to(episode_dir)),
        "combined_action_file": combined_file.name if combined_written else None,
        "original_action_backup": (
            backup_delta_file.name if backup_delta_file.is_file() else None
        ),
        "legacy_combined_action_file": (
            legacy_combined_file.name
            if legacy_combined_file.is_file() else None
        ),
        "legacy_combined_action_file_removed": (
            args.legacy_combined_name if legacy_combined_removed else None
        ),
        "combined_action_warning": combined_warning,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.plot:
        save_plot(plot_file, raw, smoothed, commands, transitions)
    if combined_warning:
        print(f"  WARNING: combined action file not written: {combined_warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive latched 1=open / 0=close absolute gripper actions."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="One episode or a dataset root containing ee_poses.npy files.",
    )
    parser.add_argument("--write", action="store_true", help="Write output files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument("--plot", action="store_true", help="Save a diagnostic PNG per episode.")
    parser.add_argument(
        "--gripper-column", type=int, default=6, help="Closure column in ee_poses.npy."
    )
    parser.add_argument(
        "--closure-direction",
        choices=("increasing", "decreasing"),
        default="increasing",
        help="Whether larger measured values mean a more closed hand.",
    )
    parser.add_argument(
        "--initial-command",
        choices=("auto", "open", "closed"),
        default="auto",
        help="Initial latched state. Auto compares the beginning with episode percentiles.",
    )
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--motion-window", type=int, default=5)
    parser.add_argument("--confirmation", type=int, default=2)
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=None,
        help="Absolute cumulative motion needed for a transition; default is data-adaptive.",
    )
    parser.add_argument("--threshold-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-threshold", type=float, default=0.003)
    parser.add_argument("--min-gap", type=int, default=10)
    parser.add_argument(
        "--manual-transitions",
        default=None,
        help="Override detection, e.g. 'close:180,open:932'. Use on one episode at a time.",
    )
    parser.add_argument(
        "--output-dir-name",
        default="gripper",
        help="Episode-relative folder for command and conversion diagnostics.",
    )
    parser.add_argument(
        "--output-name", default="gripper_actions_absolute.npy"
    )
    parser.add_argument(
        "--combined-output-name", default="ee_deltas.npy"
    )
    parser.add_argument(
        "--backup-delta-name", default="ee_deltas_not_absolute_gripper.npy"
    )
    parser.add_argument(
        "--legacy-combined-name", default="ee_actions_absolute_gripper.npy"
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not create the combined 7D action array.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(args.input_path)
    episode_dirs = find_episode_dirs(args.input_path)
    if not episode_dirs:
        raise FileNotFoundError(f"No ee_poses.npy files found under {args.input_path}")
    if args.manual_transitions and len(episode_dirs) != 1:
        raise ValueError("--manual-transitions can only be used with one episode")

    for episode_dir in episode_dirs:
        convert_episode(episode_dir, args)

    mode = "Wrote" if args.write else "Dry run for"
    print(f"\n{mode} {len(episode_dirs)} episode(s).")
    if not args.write:
        print("Review the transitions, then rerun with --write (and preferably --plot).")


if __name__ == "__main__":
    main()
