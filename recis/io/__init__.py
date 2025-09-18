from recis.info import is_internal_enabled


if is_internal_enabled():
    from recis.io.lake_dataset import LakeStreamDataset as LakeStreamDataset
    from recis.io.odps_dataset import OdpsDataset as OdpsDataset
    from recis.io.window_io import (
        make_lake_stream_window_io as make_lake_stream_window_io,
        make_odps_window_io as make_odps_window_io,
    )

    __all__ = [
        "LakeStreamDataset",
        "OdpsDataset",
        "make_lake_stream_window_io",
        "make_odps_window_io",
    ]

else:
    from recis.io.orc_dataset import OrcDataset as OrcDataset

    __all__ = ["OrcDataset"]
