def setup():
    from robustot.attackers.IoUAtk import IoUAtkConfig, IoUAtkProcessor

    iou_config = IoUAtkConfig()

    return IoUAtkProcessor("iou_atk", iou_config)
