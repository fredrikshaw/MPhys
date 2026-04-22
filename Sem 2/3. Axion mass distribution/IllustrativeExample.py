# We start by importing fetch_polytopes,
# a plotting package, and numpy
from cytools import fetch_polytopes
import matplotlib.pyplot as plt
import numpy as np

# These are the settings for the scan. 
# We scan h11=2,3,4,5,10,15,...,100
# For each h11 we take 25 polytopes
h11s = [2,3,4] + list(range(5,105,5))
n_polys = 25

# These are the lists where we will save the data
h11_list = []
nonzerointnums = []
costhetamin = []
dmins = []
Xvols = []

for h11 in h11s:
    print(f"Processing h11={h11}", end="\r")
    for p in fetch_polytopes(h11=h11, lattice="N",
                             favorable=True, limit=n_polys):
        # Here we use a single triangulation constructed using topcom,
        # to more closely reproduce the data in the paper.
        t = p.triangulate(backend="topcom")
        cy = t.get_cy()
        h11_list.append(h11)
        nonzerointnums.append(len(cy.intersection_numbers(in_basis=True)))
        mori_rays = cy.toric_mori_cone(in_basis=True).rays()
        mori_rays_norms = np.linalg.norm(mori_rays, axis=1)
        n_mori_rays = len(mori_rays)
        costhetamin.append(min(
            mori_rays[i].dot(mori_rays[j])
                /(mori_rays_norms[i]*mori_rays_norms[j])
            for i in range(n_mori_rays) for j in range(i+1,n_mori_rays)))
        tip = cy.toric_kahler_cone().tip_of_stretched_cone(1)
        dmins.append(np.log10(np.linalg.norm(tip)))
        Xvols.append(np.log10(cy.compute_cy_volume(tip)))
print("Finished processing all h11s!")
print(f"Scanned through {len(h11_list)} CY hypersurfaces.")

# We plot the data using matplotlib.
# If you are not familiar with this package, you can find tutorials and
# documentation at https://matplotlib.org/

xdata = [h11_list]*3 + [np.log10(h11_list)]
ydata = [nonzerointnums, costhetamin, dmins, Xvols]
xlabels = [r"$h^{1,1}$"]*3 + [r"log${}_{10}(h^{1,1})$"]
ylabels = [r"# nonzero $\kappa_{ijk}$", r"$\cos(\theta_{min})$",
           r"log${}_{10}(d_{min})$", r"log${}_{10}(\mathcal{V})$"]
fig, ax0 = plt.subplots(2, 2, figsize=(15,12))

for i,d in enumerate(ydata):
    ax = plt.subplot(221+i)
    ax.scatter(xdata[i], ydata[i], s=10)
    plt.xlabel(xlabels[i], size=20)
    plt.ylabel(ylabels[i], size=20)
    plt.tick_params(labelsize=15, width=2, length=5)

plt.subplots_adjust(wspace=0.3, hspace=0.22)

# Save the figure
plt.savefig('cy_hypersurface_properties.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'cy_hypersurface_properties.png'")

# Display the figure
plt.show()