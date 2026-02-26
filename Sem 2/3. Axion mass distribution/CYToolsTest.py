from cytools import read_polytopes, fetch_polytopes

g = fetch_polytopes(h21=7, lattice="N", limit = 100)
print(g)
print("\n")

p = g[0]
t = p.triangulate
print(t)