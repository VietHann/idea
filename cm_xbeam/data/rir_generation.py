"""Reverberant moving-source simulation with emission-time RIR interpolation."""
import hashlib
import importlib.metadata
import math
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

SCENARIOS = ['stationary', 'motion', 'jump', 'crossing', 'dual_movement', 'target_silence', 'approach']


def trajectories(times, scenario, duration, velocity=15, jump_degrees=60, separation=40, base_angle=0):
    tau = times - duration / 2
    first = np.full_like(times, base_angle - separation / 2)
    second = np.full_like(times, base_angle + separation / 2)
    if scenario in {'motion', 'target_silence'}:
        first += velocity * times
    elif scenario == 'dual_movement':
        first += velocity * times
        second -= velocity * times
    elif scenario == 'jump':
        first[times >= duration / 2] += jump_degrees
    elif scenario == 'crossing':
        # Identities actually cross in azimuth; actual minimum angle is near 0,
        # independently of the nominal initial angular separation.
        first = base_angle + velocity * tau
        second = base_angle - velocity * tau
    elif scenario == 'approach':
        gap = np.sqrt((velocity * tau) ** 2 + (separation / 2) ** 2)
        first, second = base_angle - gap, base_angle + gap
    elif scenario != 'stationary':
        raise ValueError(f'Unknown motion scenario {scenario}')
    return np.stack([first, second])


def load_mono(path, sample_rate, samples):
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f'Clean source must be mono: {path}')
    audio = audio[:, 0]
    if rate != sample_rate:
        common = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // common, rate // common).astype(np.float32)
    audio = audio[:samples]
    if len(audio) < samples:
        audio = np.pad(audio, (0, samples - len(audio)))
    peak = np.abs(audio).max()
    if peak <= 1e-8:
        raise ValueError(f'Silent clean source: {path}')
    return audio * (0.3 / peak)


def room_rirs(room_dimensions, microphones, positions, sample_rate, rt60, max_order=None):
    try:
        import pyroomacoustics as pra
    except ImportError as exc:
        raise ImportError('Dynamic generation requires real pyroomacoustics RIRs; install cm-xbeam[simulation] in a project-local environment') from exc
    absorption, suggested_order = pra.inverse_sabine(rt60, room_dimensions)
    order = suggested_order if max_order is None else max_order
    room = pra.ShoeBox(room_dimensions, fs=sample_rate, materials=pra.Material(absorption), max_order=order)
    room.add_microphone_array(microphones.T)
    for position in positions:
        room.add_source(position)
    room.compute_rir()
    return room.rir, order


def interpolate_convolution(dry, impulse_responses, knots, samples, abrupt=False, transition=None):
    # impulse_responses[node][microphone] for one source. RIR changes with the
    # emission time, while delayed reflections continue beyond their input block.
    microphones = len(impulse_responses[0])
    longest = max(len(r) for node in impulse_responses for r in node)
    output = np.zeros((microphones, samples + longest), dtype=np.float64)
    for i in range(len(knots) - 1):
        start, stop = int(knots[i]), int(knots[i + 1])
        if stop <= start:
            continue
        segment = dry[start:stop]
        if abrupt:
            fraction = np.zeros(len(segment))
        else:
            fraction = np.arange(len(segment), dtype=np.float64) / max(1, stop - start)
        for microphone in range(microphones):
            a = fftconvolve(segment * (1 - fraction), impulse_responses[i][microphone])
            b = fftconvolve(segment * fraction, impulse_responses[i + 1][microphone]) if not abrupt else np.zeros_like(a)
            output[microphone, start:start + len(a)] += a
            output[microphone, start:start + len(b)] += b
    return output[:, :samples].astype(np.float32)


def simulate_example(row, sample_rate=16000, microphones=3, duration=2.0, scenario='motion',
                     velocity=15, jump_degrees=60, separation=40, seed=0, rir_interval=0.1,
                     rt60_range=(0.1, 0.8), snr_range=(-5, 20), microphone_jitter=0.0, max_order=None):
    if scenario not in SCENARIOS or duration <= 0 or rir_interval <= 0:
        raise ValueError('Invalid trajectory/duration/RIR interval')
    source_files = row.get('metadata', {}).get('source_audio')
    if source_files is None or len(source_files) != 2:
        raise ValueError('Dynamic two-speaker simulation requires two clean source_audio paths in the input manifest')
    digest = hashlib.sha256(f'{seed}:{row["id"]}:{scenario}:{velocity}:{jump_degrees}:{separation}'.encode()).digest()
    example_seed = int.from_bytes(digest[:8], 'little') % 2**32
    rng = np.random.default_rng(example_seed)
    np.random.seed(example_seed)
    samples = round(duration * sample_rate)
    dry = np.stack([load_mono(path, sample_rate, samples) for path in source_files])
    if scenario == 'target_silence':
        lo, hi = round(.4 * samples), round(.6 * samples)
        dry[0, lo:hi] = 0
    dimensions = rng.uniform([4, 4, 2.5], [8, 9, 3.5])
    center = dimensions / 2
    center[2] = min(1.5, dimensions[2] / 2)
    rotation = rng.uniform(0, 2 * np.pi)
    phi = rotation + np.arange(microphones) * 2 * np.pi / microphones
    array = center[None] + np.stack([.05 * np.cos(phi), .05 * np.sin(phi), np.zeros(microphones)], axis=-1)
    if microphone_jitter:
        array += rng.normal(0, microphone_jitter, array.shape)
    distances = rng.uniform(.8, 1.2, 2)
    # Distinct radii avoid physical source collision even at equal azimuth.
    if scenario == 'crossing':
        distances = np.array([.8, 1.2])
    rt60 = float(rng.uniform(*rt60_range))
    knots = np.unique(np.r_[np.arange(0, samples, max(1, round(rir_interval * sample_rate))), samples, samples // 2]).astype(int)
    angles = trajectories(knots / sample_rate, scenario, duration, velocity, jump_degrees, separation,
                          base_angle=float(rng.uniform(-90, 90)))
    responses = [[], []]
    cache = {}
    orders = []
    for node in range(len(knots)):
        key = tuple(np.round(angles[:, node], 8))
        if key not in cache:
            rad = np.deg2rad(angles[:, node]) + rotation
            positions = center[None] + np.stack([distances * np.cos(rad), distances * np.sin(rad), np.zeros(2)], axis=-1)
            rir, order = room_rirs(dimensions, array, positions, sample_rate, rt60, max_order)
            cache[key] = (rir, order)
        rir, order = cache[key]
        orders.append(order)
        for source in range(2):
            responses[source].append([np.asarray(rir[m][source]) for m in range(microphones)])
    images = np.stack([interpolate_convolution(dry[k], responses[k], knots, samples, abrupt=scenario == 'jump') for k in range(2)])
    mixture = images.sum(0)
    snr = float(rng.uniform(*snr_range))
    noise = rng.standard_normal(mixture.shape).astype(np.float32)
    noise *= np.sqrt(np.mean(mixture ** 2) / max(1e-8, np.mean(noise ** 2)) / (10 ** (snr / 10)))
    mixture += noise
    dense_times = np.arange(0, samples + 1, max(1, round(sample_rate * .01))) / sample_rate
    # All trajectory equations are affine in their base angle; use the exact
    # same base at RIR knots and on the stored dense DoA grid.
    original_dense = trajectories(dense_times, scenario, duration, velocity, jump_degrees, separation, base_angle=0)
    original_knots = trajectories(knots / sample_rate, scenario, duration, velocity, jump_degrees, separation, base_angle=0)
    base = float(angles[0, 0] - original_knots[0, 0])
    dense = original_dense + base
    gap = np.abs((dense[0] - dense[1] + 180) % 360 - 180)
    metadata = {'source_audio': source_files, 'speaker_ids': row['metadata'].get('speaker_ids', []),
                'room_dimensions': dimensions.tolist(), 'array_center': center.tolist(), 'microphone_positions': array.tolist(),
                'source_distance_m': distances.tolist(), 'array_rotation_rad': rotation, 'rt60': rt60,
                'snr_db': snr, 'motion_scenario': scenario, 'angular_velocity': 0 if scenario == 'stationary' else velocity,
                'jump_degrees': jump_degrees if scenario == 'jump' else 0,
                'angular_separation': float(gap.min()), 'nominal_separation': separation,
                'sample_seed': example_seed, 'room_partition': row['split'], 'room_id': f'{row["split"]}:{example_seed}',
                'rir_interval_seconds': rir_interval, 'rir_max_order': max(orders),
                'rir_engine': 'pyroomacoustics image-source', 'pyroomacoustics_version': importlib.metadata.version('pyroomacoustics'),
                'rir_interpolation': 'linear in emission time; jump holds piecewise-constant RIR',
                'target_silence_interval_seconds': [.4 * duration, .6 * duration] if scenario == 'target_silence' else None}
    return mixture, images, dense.astype(np.float32), metadata
