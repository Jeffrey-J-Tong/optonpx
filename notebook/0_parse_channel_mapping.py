#%%
import sys
sys.path.append("..")
from src import build_npx2_multishank_channel_mapping, build_npx2_multishank_channel_structure, print_structure_summary

mapping   = build_npx2_multishank_channel_mapping()
structure = build_npx2_multishank_channel_structure(mapping)
print_structure_summary(structure)
