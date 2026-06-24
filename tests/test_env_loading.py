import os

from calls_analyser.env import load_environment


def test_load_environment_strips_utf8_bom_from_first_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\ufeffGOOGLE_APPLICATION_CREDENTIALS=C:\\keys\\service-account.json\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    assert load_environment(env_file) is True

    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "C:\\keys\\service-account.json"
