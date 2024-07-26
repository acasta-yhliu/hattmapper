import matplotlib.pyplot as plt

plt.rc("font", size=32, family="serif", serif="cmr10")
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['mathtext.fontset'] = 'cm'

jw = ("JW", -1.3209011193140847, 0.3117595321527502)
bk = ("BK", -1.2588759249221313, 0.43528868011844934)
btt = ("BTT", -1.0589417409162922, 0.1347010014665952)
fh = ("FH", -1.4167123485497235, 0.3222858718976118)
opttree = ("HATT", -1.3326066364085893, 0.35421429872540877)

THEO = -1.857275030202381


plt.figure(figsize=(13, 5), layout="constrained")
for n, (mapper, exp, var) in enumerate((jw, bk, btt, fh, opttree)):
    plt.plot([n - 0.3, n + 0.3], [exp, exp], color="C3", linewidth=5)
    plt.text(n - 0.46, exp + 0.03, f"{exp:.3f}", color="C3", fontsize=28)
    # plt.errorbar(n, exp, var, color="black", linewidth=3)
    # plt.text(n + 0.05, exp - var, f"{var:.3f}", fontsize=28)

plt.xlim(0 - 0.5, 4 + 0.5)
plt.ylim(THEO - 0.3, -0.8)
plt.yticks([-2.0, -1.6, -1.2, -0.8])
plt.ylabel("Energy")
plt.xticks([0, 1, 2, 3, 4], ["JW", "BK", "BTT", "FH", "HATT"], fontfamily="sans serif")

plt.plot([-1, 5], [THEO, THEO], linewidth=3)
plt.annotate(f"THEORETICAL={THEO:.3f}", (1.8, THEO - 0.01), (2, THEO - 0.28), arrowprops=dict(facecolor='C0', edgecolor="C0", shrink=0.05, width=2), fontfamily="monospace", color="C0")

plt.savefig("realsystem.pdf")