import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#with np.load('results/ensamble_simulations_spd_relax=5.npz') as results:
with np.load('results/ensamble_simulations_spd_relax=8.npz') as results:

    # square the norm difference
    norm_diffs2 = results['norm_diffs'] ** 2
    obj_poly_vals = results['obj_poly_vals']


print("Number of simulations that did not converge = {}".format(np.isnan(norm_diffs2).sum()))

data = {
    "log_norm_diffs2" : np.log10(norm_diffs2),
    "log_obj_poly_vals" : np.log10(obj_poly_vals),
}

g = sns.jointplot(data=data, x="log_obj_poly_vals", y="log_norm_diffs2")

plt.show()