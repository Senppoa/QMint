import unittest
from pathlib import Path
from unittest.mock import patch

from qmint.cli import _parser, main
from qmint.tui import _gpu_argument, _required_gpu_workers, run as run_tui


class CliAndTuiTests(unittest.TestCase):
    def test_qmint_without_subcommand_opens_tui(self):
        with (
            patch("qmint.cli.config_path", return_value=Path("/missing/config.json")),
            patch("qmint.tui.run") as run,
        ):
            self.assertEqual(main([]), 0)
        run.assert_called_once_with(first_run=True)

    def test_removed_command_aliases_are_rejected(self):
        parser = _parser()
        for arguments in (["exit"], ["switch", "uma-s"], ["start", "--np", "2"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                parser.parse_args(arguments)

    def test_start_short_options_are_supported(self):
        args = _parser().parse_args(
            [
                "start",
                "-m",
                "uma-s",
                "-b",
                "fairchem",
                "-n",
                "2",
                "-g",
                "0",
                "-d",
            ]
        )
        self.assertEqual(
            (args.model, args.backend, args.workers, args.gpu, args.debug),
            ("uma-s", "fairchem", 2, "0", True),
        )

    def test_gpu_arguments_cover_cpu_single_and_multiple_cards(self):
        self.assertIsNone(_gpu_argument("cpu", "auto"))
        self.assertEqual(_gpu_argument("single", "2,3"), "2")
        self.assertEqual(_gpu_argument("multi", "0,2-3"), "0,2-3")
        self.assertEqual(_required_gpu_workers("multi", "0,2-3"), 3)

    def test_tui_exit_stops_workers_to_release_accelerators(self):
        def open_and_start(callback):
            with patch("qmint.tui._draw") as draw:
                draw.side_effect = lambda screen, first_run, runtime: runtime.update(
                    server_started=True
                )
                callback(None)

        with (
            patch("qmint.tui.curses.wrapper", side_effect=open_and_start),
            patch("qmint.tui.server_info", return_value={"pid": 123}),
            patch("qmint.tui.stop_server") as stop,
        ):
            run_tui(first_run=False)
        stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
