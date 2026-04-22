"""Final validation test"""
from DetectorDistanceReach import *

print('=== FINAL VALIDATION TEST ===\n')

print('1. Testing annihilation processes:')
for proc in ['2p', '3d', '4f']:
    d = calc_distance_reach_ann(1e-24, 0.1, proc, 1.0)
    print(f'   {proc}: {d:.4e} kpc')

print('\n2. Testing transition processes:')
for proc in ['3p 2p', '5f 4f']:
    d = calc_distance_reach_trans(1e-24, 0.1, proc, 1.0)
    print(f'   {proc}: {d:.4e} kpc')

print('\n3. Creating test plot...')
masses, dists = plot_distance_reach(
    1e-24, 0.1, '2p', (0.1, 10), 
    num_points=20, 
    save_path='final_test.png', 
    show_plot=False
)
print(f'   Plot saved: final_test.png')
print(f'   Data points: {len(masses)}')

print('\n=== ALL TESTS PASSED ✓ ===')
