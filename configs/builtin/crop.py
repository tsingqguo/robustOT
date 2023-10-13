def setup():
    from msot.tools.test.utils.process.builtin import (
        DefaultCrop,
        DefaultCropConfig,
    )

    return DefaultCrop(None, DefaultCropConfig())
