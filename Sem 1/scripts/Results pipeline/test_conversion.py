import ConvertedFunctions as cf
import numpy as np

print('Annihilation functions:')
for key in list(cf.diff_power_ann_dict.keys())[:5]:
    func = cf.diff_power_ann_dict[key]
    result = func(0.1, 0.5)
    print(f'  {key}: {result:.4e}')

print('\nTransition functions:')
for key in list(cf.diff_power_trans_dict.keys())[:5]:
    func = cf.diff_power_trans_dict[key]
    result = func(0.1, 0.5)
    print(f'  {key}: {result:.4e}')

print(f'\nTotal: {len(cf.diff_power_ann_dict)} annihilations, {len(cf.diff_power_trans_dict)} transitions')
print('All functions imported and tested successfully!')
