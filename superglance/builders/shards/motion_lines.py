"""The road-users block of earlier pools' prompts ("Road users ahead from the camera ...", inserted right before the
instruction). No stage-3 row has it; superglance.builders.shards.common.strip and the RL pool cut it out of earlier rows."""

HEAD = "Road users ahead from the camera over the last 1 s"
UNITS = " (x ahead of our rear axle, y left, metres; speed along our heading relative to ours, m/s): "
INSTR = "Choose the trajectory for the next 4 seconds."   # stage3_navsim.C256_INSTRUCTION


