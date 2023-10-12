from typing import TypedDict

from .result import ResultStyle


class DefaultResultTemplate(TypedDict):
    main: str


def default_template_builder(
    template: DefaultResultTemplate,
    sequence_name: str,
    style: ResultStyle,
) -> DefaultResultTemplate:
    if style is ResultStyle.VOT_ST:
        return os.path.join(
            test_result_root,
            sequence_name,
        )
    elif style is ResultStyle.VOT_LT:
        return os.path.join(
            test_result_root,
            sequence_name,
            "longterm",
        )
    else:
        return test_result_root
