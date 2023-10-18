import torch
import numpy as np

from msot.trackers.base import TrackResult

from ..utils.roles import TDRoles
from .types import (
    ParamsFrame,
    ParamsFrameFinish,
    ParamsFrameInit,
    ParamsFrameSkip,
    ParamsFrameTrack,
)


def action_empty(params: ParamsFrame) -> None:
    ...


def action_init(params: ParamsFrameInit) -> None:
    frame = params.historical.cur.frame.unwrap()

    # TODO: role
    p_out = params.input_process.execute(frame.img.get(TDRoles.TEST))
    if isinstance(p_out.input, np.ndarray):
        params.raw_tracker.init(
            frame.img.get(TDRoles.TEST),  # TODO: role
            frame.bbox.get(TDRoles.TEST),  # TODO: role
        )
    else:
        params.raw_tracker.init_with_scaled_template(p_out.input.crop)

    tracking = params.historical.set_cur_tracking(lambda T: T())
    tracking.scaled_z = p_out.scaled_crop_vert


def action_skip(params: ParamsFrameSkip) -> None:
    params.historical.set_cur_tracking(lambda T: T())


def action_track(params: ParamsFrameTrack) -> TrackResult:
    frame = params.historical.cur.frame.unwrap()

    # TODO: role
    p_out = params.input_process.execute(frame.img.get(TDRoles.TEST))
    if isinstance(p_out.input, np.ndarray):
        _, res = params.raw_tracker.track(p_out.input)
    else:
        res = params.raw_tracker.track_with_scaled_search(
            p_out.input.crop,
            p_out.input.size.scale,
            params.sequence_info.size.val,
        )

    tracking = params.historical.set_cur_tracking(lambda T: T())
    tracking.processor_attrs.update(p_out.processor_attrs)
    tracking.scaled_z = (
        params.historical.last.unwrap().tracking.scaled_z.smart_clone()
    )
    tracking.scaled_x = p_out.scaled_crop_vert
    return res


def action_finish(params: ParamsFrameFinish) -> None:
    result = params.historical.get_historical_result()
    analysis = params.historical.get_historical_analysis()
    results = result.to_list()
    analysis = analysis.to_list()

    params.result_utils.save_results_default(
        sequence_name=params.sequence_info.name,
        results=results,
        analysis=analysis,
    )
