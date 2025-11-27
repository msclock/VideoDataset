"""Test for cli."""

from click.testing import CliRunner

from fast_context_queue.cli import cli


def test_cli() -> None:
    """Test for cli."""
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 2
    assert "Usage" in result.output


def test_cli_run() -> None:
    """Test for run subcommand of the cli."""
    runner = CliRunner()
    result = runner.invoke(cli, "run")
    assert result.exit_code == 0
    assert not result.output
