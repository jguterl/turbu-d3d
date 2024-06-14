# TEMPORARY FILE: EDITS WILL NOT BE SAVED!
# This file was generated by omfit_classes.utils_base.function_to_tree()

defaultVars(position_type='psi',
params=['temp', 'density'],
unit_convert=True,
combine_data_before_contouring=False,
num_color_levels=41,
fig=None,
axs=None,)
self = 

"""
Plots contours of a physics quantity vs. time and space

:param position_type: string
    Select position from 'R', 'Z', or 'PSI'

:param params: list of strings
    Select parameters from 'temp', 'density', 'press', or 'redchisq'

:param unit_convert: bool
    Convert units from e.g. eV to keV to try to make most quantities closer to order 1

:param combine_data_before_contouring: bool
    Combine data into a single array before calling tricontourf. This may look smoother, but it can hide the way
    arrays from different subsystems are stitched together

:param num_color_levels: int
    Number of contour levels

:param fig: Figure instance
    Provide a Matplotlib Figure instance and an appropriately dimensioned array of Axes instances to overlay

:param axs: array of Axes instances
    Provide a Matplotlib Figure instance and an appropriately dimensioned array of Axes instances to overlay

:return: Figure instance, array of Axes instances
    Returns references to figure and axes used in plot
"""
from matplotlib import pyplot

if fig is None:
    fig = pyplot.figure()
if axs is None:
    axs = self.setup_axes(fig, len(tolist(params)), sharex='all')
axs = np.atleast_1d(axs)
axs[-1].set_xlabel('Time (ms)')
axs[0].set_title('Thomson scattering')
for ax in axs[:-1]:
    ax.tick_params(labelbottom=False)

tt = np.array([])
xx = np.array([])
yy = np.array([])

names = {
    'press': '$p_e$ (kPa)' if unit_convert else '$p_e$ (Pa)',
    'density': '$n_e$ (10$^{19}$/m$^{3}$)' if unit_convert else '$n_e$ (m$^{-3}$)',
    'temp': '$T_e$ (keV)' if unit_convert else '$T_e$ (eV)',
}
multipliers = {'press': 1e-3 if unit_convert else 1, 'density': 1e-19 if unit_convert else 1, 'temp': 1e-3 if unit_convert else 1}

for ax, param in zip(axs, params):
    for sub in self.subsystems:
        okay = self['filtered'][sub]['filters']['okay']
        t = self['filtered'][sub]['time'][np.newaxis, :] + 0 * okay

        if position_type in ['z', 'Z']:
            x = self['filtered'][sub]['z'][:, np.newaxis] + 0 * okay
            ax.set_ylabel('$Z$ (m)')
        elif position_type in ['r', 'R']:
            x = self['filtered'][sub]['r'][:, np.newaxis] + 0 * okay
            ax.set_ylabel('$R$ (m)')
        else:
            x = self['filtered'][sub]['psin_TS']
            ax.set_ylabel(r'$\psi_N$')

        y = nominal_values(self['filtered'][sub][param]) * multipliers.get(param, 1)

        xf = x.flatten()
        tf = t.flatten()
        yf = y.flatten()
        w = okay.flatten().astype(bool)
        tf = tf[w]
        xf = xf[w]
        yf = yf[w]

        tt = np.append(tt, tf)
        xx = np.append(xx, xf)
        yy = np.append(yy, yf)

        if not combine_data_before_contouring:
            im = ax.tricontourf(tf, xf, yf, num_color_levels)

    if combine_data_before_contouring:
        im = ax.tricontourf(tt, xx, yy, num_color_levels)

    cb = pyplot.colorbar(im, ax=ax)
    cb.set_label(names.get(param, param))
    ax.axhline(1.0, color='k', linestyle='--')

try:
    # Getting at OMFIT plot utilities when doing stand-alone command-line stuff can be a burden
    cornernote(shot=self.shot, device=self.device, time='')
except NameError:
    fig.text(0.99, 0.01, '{}#{}'.format(self.device, self.shot), fontsize=10, ha='right', transform=pyplot.gcf().transFigure)