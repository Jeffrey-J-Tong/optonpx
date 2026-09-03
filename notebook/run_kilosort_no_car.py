from pathlib import Path
from kilosort import run_kilosort
from kilosort.io import load_probe

# 加载你自己的 .mat channel map(不是官方内置的 NeuroPix1_default.mat)
probe = load_probe('/path/to/your_channel_map.mat')

settings = {
    'filename': '/path/to/your_data.bin',
    'n_chan_bin': 385,          # 改成你实际的通道数
    'Th_universal': 10,
    'Th_learned': 8,
    'duplicate_spike_ms': 0.25,
}

ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(
        settings=settings,
        probe=probe,
        shank_idx=[0, 1, 2, 3],   # 4个shank分开跑,各自存到独立子文件夹
        do_CAR=False,             # 关闭CAR
    )