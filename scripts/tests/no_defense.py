from pyotp.tools.test_aio import (
    TestAIOArgs as _TestAIOArgs,
    get_test_interface,
    get_wrappers,
)
from pyotp.tools.test_instances import TI_NoDef


class TestAIOArgs(_TestAIOArgs):
    ...


if __name__ == "__main__":
    wrappers = get_wrappers(
        TestAIOArgs,
    )
    for wrapper in wrappers:
        ti = get_test_interface(wrapper)

        user_suffix = "_" + ti.args.snapshot_name
        if ti.wrapper_args.suffix is not None:
            user_suffix += "_" + ti.wrapper_args.suffix

        rtaa = TI_NoDef(ti.test)
        rtaa.user_suffix = user_suffix
        rtaa.run(ti.args.dataset or [], test_attr_params=ti.test_attr_params)
