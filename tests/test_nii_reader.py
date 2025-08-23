from pathlib import Path

from segmentation.helpers.nii_reader import NiiReader


def test_parse_config(tmp_path: Path):
    cfg = tmp_path / "Info.cfg"
    cfg.write_text(
        """
        ED: 5
        ES: 12
        Group: HCM
        Height: 175.5
        NbFrame: 30
        Weight: 72.0
        """.strip()
    )

    reader = NiiReader()
    parsed = reader.parse_config(str(cfg))

    assert parsed["ED"] == "5"
    assert parsed["ES"] == "12"
    assert parsed["Group"] == "HCM"
    assert parsed["Height"] == "175.5"
    assert parsed["NbFrame"] == "30"
    assert parsed["Weight"] == "72.0"

