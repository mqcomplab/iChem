import unittest
import tempfile
import json
from pathlib import Path
import numpy as np

# Import cli functions lazily to avoid import issues with bblean
# when tests directory is in sys.path
main = None
_build_parser = None

def _import_cli():
    global main, _build_parser
    if main is None:
        from iChem.cli import main as _main, _build_parser as _parser
        main = _main
        _build_parser = _parser

class TestFingerprinterCLI(unittest.TestCase):
    """Test CLI commands for fingerprint generation."""

    def setUp(self):
        """Set up temporary directory for test outputs."""
        _import_cli()  # Lazy import to avoid pytest collection issues
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_binary_fps_cli_default(self):
        """Test binary-fps CLI with default parameters."""
        output_file = self.temp_path / "test_binary_fps.npy"
        argv = [
            "binary-fps",
            "tests/data/molecules.smi",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        self.assertTrue(output_file.exists())
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)
        self.assertEqual(fps.shape[1], 2048 // 8)  # packed by default

    def test_binary_fps_cli_unpacked(self):
        """Test binary-fps CLI with unpacked output."""
        output_file = self.temp_path / "test_binary_fps_unpacked.npy"
        argv = [
            "binary-fps",
            "tests/data/molecules.smi",
            "--packed", "false",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        self.assertTrue(output_file.exists())
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)
        self.assertEqual(fps.shape[1], 2048)  # unpacked

    def test_binary_fps_cli_maccs(self):
        """Test binary-fps CLI with MACCS fingerprint type."""
        output_file = self.temp_path / "test_binary_fps_maccs.npy"
        argv = [
            "binary-fps",
            "tests/data/molecules.smi",
            "--fp-type", "MACCS",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        self.assertTrue(output_file.exists())
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)
        self.assertEqual(fps.shape[1], 167)  # MACCS fixed size

    def test_binary_fps_cli_with_invalid_smiles(self):
        """Test binary-fps CLI with return-invalid flag."""
        output_file = self.temp_path / "test_binary_fps_invalid.npy"
        argv = [
            "binary-fps",
            "tests/data/molecules_invalids.smi",
            "--return-invalid", "true",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        self.assertTrue(output_file.exists())
        
        # Check .npy file
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)
        
        # Check invalid indices JSON file
        invalid_file = self.temp_path / "test_binary_fps_invalid_invalid_indices.json"
        self.assertTrue(invalid_file.exists())
        with open(invalid_file) as f:
            invalid_indices = json.load(f)
        self.assertEqual(len(invalid_indices), 3)
        self.assertListEqual(invalid_indices, [6, 65, 120])

    def test_count_fps_cli_default(self):
        """Test count-fps CLI with default parameters."""
        output_file = self.temp_path / "test_count_fps.npy"
        argv = [
            "count-fps",
            "tests/data/molecules.smi",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        self.assertTrue(output_file.exists())
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)
        self.assertEqual(fps.shape[1], 2048)

    def test_count_fps_cli_with_invalid_smiles(self):
        """Test count-fps CLI with return-invalid flag."""
        output_file = self.temp_path / "test_count_fps_invalid.npy"
        argv = [
            "count-fps",
            "tests/data/molecules_invalids.smi",
            "--return-invalid", "true",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        self.assertTrue(output_file.exists())
        
        # Check .npy file
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)
        
        # Check invalid indices JSON file
        invalid_file = self.temp_path / "test_count_fps_invalid_invalid_indices.json"
        self.assertTrue(invalid_file.exists())
        with open(invalid_file) as f:
            invalid_indices = json.load(f)
        self.assertEqual(len(invalid_indices), 3)
        self.assertListEqual(invalid_indices, [6, 65, 120])

    def test_real_fps_cli_default(self):
        """Test real-fps CLI with default parameters."""
        output_file = self.temp_path / "test_real_fps.npy"
        argv = [
            "real-fps",
            "tests/data/molecules.smi",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        self.assertTrue(output_file.exists())
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)
        self.assertEqual(fps.shape[1], 217)

    def test_real_fps_cli_with_invalid_smiles(self):
        """Test real-fps CLI with return-invalid flag."""
        output_file = self.temp_path / "test_real_fps_invalid.npy"
        argv = [
            "real-fps",
            "tests/data/molecules_invalids.smi",
            "--return-invalid", "true",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        self.assertTrue(output_file.exists())
        
        # Check .npy file
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)
        
        # Check invalid indices JSON file
        invalid_file = self.temp_path / "test_real_fps_invalid_invalid_indices.json"
        self.assertTrue(invalid_file.exists())
        with open(invalid_file) as f:
            invalid_indices = json.load(f)
        self.assertEqual(len(invalid_indices), 3)
        self.assertListEqual(invalid_indices, [6, 65, 120])

    def test_binary_fps_cli_different_nbits(self):
        """Test binary-fps CLI with different n-bits parameter."""
        output_file = self.temp_path / "test_binary_fps_1024.npy"
        argv = [
            "binary-fps",
            "tests/data/molecules.smi",
            "--n-bits", "1024",
            "--packed", "false",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)
        self.assertEqual(fps.shape[1], 1024)

    def test_count_fps_cli_different_nbits(self):
        """Test count-fps CLI with different n-bits parameter."""
        output_file = self.temp_path / "test_count_fps_1024.npy"
        argv = [
            "count-fps",
            "tests/data/molecules.smi",
            "--n-bits", "1024",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)
        self.assertEqual(fps.shape[1], 1024)

    def test_parser_missing_required_command(self):
        """Test that parser requires a command."""
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_binary_fps_cli_with_standardization(self):
        """Test binary-fps CLI with standardization enabled."""
        output_file = self.temp_path / "test_binary_fps_std.npy"
        argv = [
            "binary-fps",
            "tests/data/molecules.smi",
            "--standarize", "true",
            "--packed", "false",
            "--out", str(output_file),
        ]
        
        result = main(argv)
        
        self.assertEqual(result, 0)
        fps = np.load(output_file)
        self.assertEqual(fps.shape[0], 118)

            def test_rewrite_smiles_by_cluster_npy_cli(self):
                """Test rewriting SMILES from per-cluster .npy files."""
                smiles_dir = self.temp_path / "smiles"
                clusters_dir = self.temp_path / "clusters"
                output_dir = self.temp_path / "rewritten"
                smiles_dir.mkdir()
                clusters_dir.mkdir()

                smiles_file = smiles_dir / "molecules.smi"
                smiles = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]
                with open(smiles_file, "w") as handle:
                    for smi in smiles:
                        handle.write(f"{smi}\n")

                np.save(clusters_dir / "cluster_0.npy", np.array([0, 2, 5], dtype=np.int64))
                np.save(clusters_dir / "cluster_1.npy", np.array([1, 3, 4], dtype=np.int64))

                argv = [
                    "rewrite-smiles-by-cluster-npy",
                    "--clusters-dir", str(clusters_dir),
                    "--smiles-dir", str(smiles_dir),
                    "--output-dir", str(output_dir),
                    "--num-workers", "2",
                ]

                result = main(argv)

                self.assertEqual(result, 0)

                cluster_0 = output_dir / "cluster_0.smi"
                cluster_1 = output_dir / "cluster_1.smi"
                self.assertTrue(cluster_0.exists())
                self.assertTrue(cluster_1.exists())

                with open(cluster_0) as handle:
                    self.assertEqual(handle.read().splitlines(), ["C", "CCC", "CCCCCC"])

                with open(cluster_1) as handle:
                    self.assertEqual(handle.read().splitlines(), ["CC", "CCCC", "CCCCC"])
