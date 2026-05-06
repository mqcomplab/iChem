from pathlib import Path
import re
import subprocess

def rename_files(files_paths: list[Path]):
    """Pad trailing file numeration so paths sort naturally.

    Only supports files ending in .smi or .smi.gz; other files are
    returned unchanged.
    """
    files_paths = [Path(file_path) for file_path in files_paths]

    numbered_files = []
    for file_path in files_paths:
        name = file_path.name
        if name.endswith('.smi.gz'):
            base_name = name[:-7]
            suffix = '.smi.gz'
        elif name.endswith('.smi'):
            base_name = name[:-4]
            suffix = '.smi'
        else:
            # skip non-smi files when collecting numbers
            continue

        match = re.search(r'_(\d+)$', base_name)
        if match is None:
            continue

        digits = match.group(1)
        if digits.startswith('0'):
            continue

        numbered_files.append((file_path, base_name, suffix, digits))

    if not numbered_files:
        return files_paths

    pad_width = max(len(digits) for _, _, _, digits in numbered_files)
    renamed_files = []

    for file_path in files_paths:
        name = file_path.name
        if name.endswith('.smi.gz'):
            base_name = name[:-7]
            suffix = '.smi.gz'
        elif name.endswith('.smi'):
            base_name = name[:-4]
            suffix = '.smi'
        else:
            # leave non-smi files unchanged
            renamed_files.append(file_path)
            continue

        match = re.search(r'_(\d+)$', base_name)
        if match is None:
            renamed_files.append(file_path)
            continue

        digits = match.group(1)
        if digits.startswith('0'):
            renamed_files.append(file_path)
            continue

        padded_digits = digits.zfill(pad_width)
        new_name = f"{base_name[:match.start(1)]}{padded_digits}{suffix}"
        new_path = file_path.with_name(new_name)
        if new_path != file_path:
            file_path.rename(new_path)
        renamed_files.append(new_path)

    return renamed_files

def directory_processing(working_dir):
    # Read the files in the working directory and process them as needed
    smi_files = list(Path(working_dir).glob('*.smi')) + list(
        Path(working_dir).glob('*.smi.gz')
    )

    if not smi_files:
        raise ValueError(f"No .smi or .smi.gz files found in {working_dir}.")


def _get_file_length(file_path: Path):
    """Return number of lines in a .smi or .smi.gz file efficiently.

    Uses command-line tools (`wc -l` and `gzip -dc | wc -l`) for speed.
    """
    file_path = Path(file_path)
    file_path_str = str(file_path)

    name = file_path.name
    if name.endswith('.smi.gz'):
        gzip_proc = subprocess.Popen(
            ['gzip', '-dc', file_path_str],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        wc_proc = subprocess.run(
            ['wc', '-l'],
            stdin=gzip_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if gzip_proc.stdout is not None:
            gzip_proc.stdout.close()
        gzip_stderr = gzip_proc.communicate()[1]
        if gzip_proc.returncode != 0:
            raise RuntimeError(gzip_stderr.decode(errors='replace').strip())

        return int(wc_proc.stdout.strip().split()[0])
    elif name.endswith('.smi'):
        wc_proc = subprocess.run(
            ['wc', '-l', file_path_str],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return int(wc_proc.stdout.strip().split()[0])
    else:
        raise ValueError('Only .smi and .smi.gz files are supported')


def cluster_hpc_initial(working_dir: str,
                        threshold: float | None = None,
                        branching_factor: int = 1024,
                        merging_criterion: str = 'diameter',
                        reclustering_iterations: int = 3,
                        extra_threshold: float = 0.025,
                        fp_type: str = 'ECFP4',
                        n_bits: int = 2048,
                        output_dir: str = None,
                        ):
    """Initial clustering using BitBirch algorithm for hpc."""
    # Get the files to process
    files_paths = directory_processing(working_dir)
    files_paths = rename_files(files_paths)

    # Sort the files
    files_paths = sorted(files_paths)

    # Generate the necessary tuples for files