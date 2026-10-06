"""Render stored physical/memory diagnostics on their shared time axis."""
import argparse
import os
import numpy as np
from cm_xbeam.utils import safe_output_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='NPZ emitted by evaluate --diagnostics-dir')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    os.environ['MPLCONFIGDIR'] = str(safe_output_path('.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = [('doa_degrees', 'DoA (degrees)'), ('lambda_s', 'Target forgetting factor'),
            ('lambda_i', 'Interference forgetting factor'), ('alpha', 'Residual gate'),
            ('uncertainty', 'Predicted log SCM error'), ('covariance_error', 'Normalized oracle SCM error'),
            ('log_innovation', 'Log covariance innovation'), ('distortionless_error', 'Estimated distortionless error')]
    with np.load(args.input, allow_pickle=False) as data:
        selected = [(key, label) for key, label in keys if key in data]
        figure, axes = plt.subplots(len(selected), 1, sharex=True, figsize=(11, 2 * len(selected)), squeeze=False)
        for axis, (key, label) in zip(axes[:, 0], selected):
            values = data[key]
            for source, series in enumerate(values):
                axis.plot(data['time_seconds'], series, label=f'Source slot {source + 1}')
            axis.set_ylabel(label)
            axis.grid(alpha=.25)
        axes[0, 0].legend()
        axes[-1, 0].set_xlabel('Time (seconds; current frame acquisition end)')
        figure.tight_layout()
        output = safe_output_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)
        print(str(output))


if __name__ == '__main__':
    main()
