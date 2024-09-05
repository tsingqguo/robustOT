import os
from pyotp.tools.test import Test, TestArgs, ValidDatasetNames
from pyotp.tools.test_instances import GroupedProcessors
from typing import Any, Generic, Literal, NamedTuple, Optional, Type, TypeVar
from typed_cap import Cap


class TestAIOArgs:
    # @alias=A
    attack: str

    # @alias=d
    datasets: list[ValidDatasetNames]

    # @alias=t
    tracker: list[Literal["r50", "mob"]]

    # @alias=V
    visualization: bool = False

    # @alias=s
    suffix: Optional[str] = None

    # @alias=g
    group_name: Optional[str] = None

    # @alias=f
    force: bool = False


# TA = TypeVar("TA", bound=TestArgs)
TC = TypeVar("TC", bound=TestAIOArgs)


class TestWrapperArgs(Generic[TC]):
    wrapper_args: TC
    argv: list[str]
    datasets: list[ValidDatasetNames]

    def __init__(
        self,
        args: TC,
        argv: list[str],
        datasets: list[ValidDatasetNames],
    ):
        self.wrapper_args = args
        self.argv = argv
        self.datasets = datasets


class TestInterface(Generic[TC]):
    wrapper_args: TC
    args: TestArgs
    test: Test
    test_attr_params: dict
    group: Optional[GroupedProcessors] = None
    group_name: Optional[str]

    def __init__(
        self,
        wrapper_args: TC,
        args: TestArgs,
        test: Test,
        test_attr_params: dict,
        group: Optional[GroupedProcessors] = None,
        group_name: Optional[str] = None,
    ):
        self.wrapper_args = wrapper_args
        self.args = args
        self.test = test
        self.test_attr_params = test_attr_params
        self.group = group
        self.group_name = group_name


def built_args(
    test_aio_args: Type[TC],
    argv_overwrite: list[str] | None = None,
):
    cap = Cap(test_aio_args)
    if argv_overwrite is not None:
        parsed = cap.parse(argv=argv_overwrite)
    else:
        parsed = cap.parse()
    wrapper_args = parsed.val
    wrapper_argv = parsed.args

    if wrapper_args.attack == "none":
        dir_word = "pysot"
    elif wrapper_args.attack == "iou":
        dir_word = "IoUAtk"
    else:
        dir_word = wrapper_args.attack.lower().split(",")[0]
    if dir_word.lower() not in os.getcwd().lower().split(os.path.sep):
        print("dir_word:", dir_word)
        print(os.getcwd().lower().split(os.path.sep))
        print("YOU ARE USING A WRONG EXP FOLDER, aborting...")
        exit(1)

    wrappers: list[TestWrapperArgs[TC]] = []

    for tracker in wrapper_args.tracker:
        for dataset in wrapper_args.datasets:
            if tracker == "r50":
                pysot_pt_fp = "$PYOTP_EXP/pysot/pretrained/siamrpn_r50_l234_dwxcorr{}/model.pth"
                output_dir = "results_r50"
            elif tracker == "mob":
                pysot_pt_fp = "$PYOTP_EXP/pysot/pretrained/siamrpn_mobilev2_l234_dwxcorr/model.pth"
                output_dir = "results_mob"
            else:
                raise ValueError(f"Unknown tracker: {tracker}")

            # print("[debug] dataset:", dataset)
            # print("[debug] tracker:", tracker)
            if dataset == "OTB100":
                pysot_pt_fp = pysot_pt_fp.format("_otb")
            else:
                pysot_pt_fp = pysot_pt_fp.format("")
            # print("[debug] snapshot:", pysot_pt_fp)

            name = "pysot_pt"
            argv = [
                "-s",
                os.path.expandvars(pysot_pt_fp),
                "-n",
                name,
                "-o",
                output_dir,
                "-d",
                dataset,
            ]
            if wrapper_args.visualization:
                argv.append("-V")

            if len(wrapper_argv) > 0:
                argv.extend(wrapper_argv)

            if wrapper_args.force:
                argv.append("-f")

            wrappers.append(
                TestWrapperArgs(
                    args=wrapper_args,
                    argv=argv,
                    datasets=[dataset],
                )
            )
        # exit(1)

    return wrappers


def get_wrappers(
    test_aio_args: Type[TC],
    argv_overwrite: list[str] | None = None,
):
    return built_args(test_aio_args, argv_overwrite)


def get_test_interface(
    wrapper: TestWrapperArgs[TC],
) -> TestInterface[TC]:
    from pyotp.env import ENV
    from pyotp.tools.test import args_parse

    wrapper_args = wrapper.wrapper_args
    argv = wrapper.argv

    atk = wrapper_args.attack.split(",")
    if atk[0].lower() == "csa":
        from attackers.csa.test import (
            initial_csa_tester,
            initial_testing_args,
            TestArgs,
            TestCliArgs,
        )

        argv = [*argv, "--attack-on", "TS", "--attack-type", "cs"]

        argv, cliargs = args_parse(TestCliArgs, argv=argv)
        args = TestArgs()

        initial_testing_args(cliargs, argv, args)
        test, tracker = initial_csa_tester(args)
        test_attr_params = {}
    elif atk[0].lower() == "iou":
        from attackers.iou.test import (
            initial_testing_args,
            initial_iou_tester,
            TestArgs,
            TestCliArgs,
        )

        argv, cliargs = args_parse(TestCliArgs, argv=argv)
        args = TestArgs()

        initial_testing_args(cliargs, argv, args)
        test, tracker = initial_iou_tester(args)
        test_attr_params = {"iou_tracker": tracker}
    elif atk[0].lower() == "rtaa":
        from attackers.rtaa.test import (
            initial_testing_args,
            initial_rtaa_tester,
            TestArgs,
            TestCliArgs,
        )

        argv = [*argv, "--iteration", str(3)]

        argv, cliargs = args_parse(TestCliArgs, argv=argv)
        args = TestArgs()

        initial_testing_args(cliargs, argv, args)
        test, tracker = initial_rtaa_tester(args)
        test_attr_params = {}
    elif atk[0].lower() == "spark":
        from attackers.spark.test import (
            initial_tester,
            initial_testing_args,
            AtkType,
            TestArgs,
            TestCliArgs,
        )

        if len(atk) == 1:
            print("panic: unspecified attack type (UA/TA)")

        argv = [*argv, "--attack-type", atk[1], "--apts"]

        argv, cliargs = args_parse(TestCliArgs, argv=argv)
        args = TestArgs()

        initial_testing_args(cliargs, argv, args)
        test, oim_attacker = initial_tester(args)

        test_attr_params = {
            "attacker": oim_attacker,
            "atk_type": args.atk_type,
            "attacked_root": os.path.join(
                ENV.experiments_path,
                "SPARK",
                "attacked",
                f"test_attacked{'_ta' if args.atk_type is AtkType.TA else '' }",
            ),
        }

    elif atk[0].lower() == "none":
        from pyotp.tools.test import (
            Test,
            TestArgs,
            TestCliArgs,
            args_parse,
            initial_testing_args,
        )

        argv, cliargs = args_parse(TestCliArgs, argv=argv)
        args = TestArgs()

        initial_testing_args(cliargs, argv, args)
        test = Test(args)
        test.initial_tracker()
        test_attr_params = {}
    else:
        print("panic: unknown attack", wrapper_args.attack)
        exit(1)

    if wrapper_args.group_name is not None:
        group = {}
        group_name = wrapper_args.group_name
    else:
        group = None
        group_name = None

    ti = TestInterface(
        wrapper_args,
        args,
        test,
        test_attr_params,
        group,
        group_name,
    )
    return ti
