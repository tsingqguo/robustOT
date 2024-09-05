import os
from pyotp.env import ENV
from pyotp.tools.test_aio import (
    TestAIOArgs as _TestAIOArgs,
    get_test_interface,
    get_wrappers,
)
from pyotp.tools.test_instances import TI_NoDef, TI_LRR


class TestAIOArgs(_TestAIOArgs):
    ...


if __name__ == "__main__":
    wrappers = get_wrappers(TestAIOArgs)

    for wrapper in wrappers:
        ti = get_test_interface(wrapper)

        user_suffix = "_" + ti.args.snapshot_name
        if ti.wrapper_args.suffix is not None:
            user_suffix += "_" + ti.wrapper_args.suffix

        lrr = TI_LRR(ti.test)
        lrr.group_spawn_mode = ti.group_name is not None
        lrr.user_suffix = user_suffix
        lrr.args_FRAME_CNT = [5]
        lrr.args_saved = {
            "pretrained": os.path.expandvars(
                "$PYOTP_EXP/lrr_saves/lrr_pretrained/lrr-epoch-best.pth"
            ),
        }
        lrr.run(
            ti.args.dataset or [],
            test_attr_params=ti.test_attr_params,
            groups=ti.group,
        )

        if ti.group is not None and ti.group_name is not None:
            ti_nodef = TI_NoDef(ti.test)
            ti_nodef.user_suffix = user_suffix
            ti_nodef.group_spawn_mode = ti.group_name is not None
            ti_nodef.run([], groups=ti.group)

            for grp_name in ti.group.keys():
                print(f"Group {grp_name}:")
            print(f"tot: {len(ti.group)} groups")
            ti_nodef._run_groups(
                ti.args.dataset or [],
                grouped_processors=ti.group,
                test_attr_params={},
                sequences=[],
                exec_extra_attrs={
                    "groups_sort_fn": None,
                },
                group_results_name=ti.group_name,
            )
