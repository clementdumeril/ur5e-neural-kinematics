"""
Genere les figures du README A PARTIR des fichiers de resultats.

Une figure dessinee a la main se desynchronise du jour ou un chiffre change, et
plus personne ne s'en apercoit. Celles-ci relisent les JSON produits par les
experiences : si une mesure bouge, il suffit de relancer ce script.

    python experiments/figures.py

Sortie : assets/fig_ablation.svg et assets/fig_benchmark.svg
"""
import json
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ASSETS = os.path.join(_ROOT, 'assets')

NOIR, GRIS, GRIS_CLAIR = '#111827', '#6b7280', '#9ca3af'
ACCENT, ACCENT_FOND = '#2563eb', '#eff6ff'
POLICE = 'Helvetica, Arial, sans-serif'


def entete(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="{POLICE}">'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>')


def texte(x, y, s, taille=14, couleur=NOIR, ancre='start', gras=False):
    g = ' font-weight="bold"' if gras else ''
    return (f'<text x="{x}" y="{y}" font-size="{taille}" fill="{couleur}" '
            f'text-anchor="{ancre}"{g}>{s}</text>')


def barre(x, y, w, h, couleur, bord=None):
    b = f' stroke="{bord}" stroke-width="2"' if bord else ''
    return f'<rect x="{x}" y="{y}" width="{max(w, 1):.1f}" height="{h}" rx="3" fill="{couleur}"{b}/>'


# =====================================================================
# Figure 1 : ce que la perte physique apporte
# =====================================================================
def figure_ablation():
    chemin = os.path.join(_ROOT, 'checkpoints', 'ablation_physics_loss.json')
    res = {r['nom']: r for r in json.load(open(chemin))['resultats']}

    sup = res['branche unique | w_phys=0']['best_pos_mm']
    phy = res['branche unique | w_phys=1']['best_pos_mm']
    gain = (sup - phy) / sup

    W, H = 860, 330
    X0, LARG = 300, 420
    echelle = LARG / sup

    p = [entete(W, H)]
    p.append(texte(40, 44, 'What does the physics term actually buy?', 19, NOIR, gras=True))
    p.append(texte(40, 68, 'Validation position error — identical data, seed, architecture and schedule; '
                           'only the loss differs.', 13, GRIS))

    for i, (nom, val, couleur, bord) in enumerate([
            ('Supervised only', sup, '#d1d5db', None),
            ('Physics-informed', phy, ACCENT, None)]):
        y = 118 + i * 74
        p.append(texte(X0 - 18, y + 30, nom, 15, NOIR, 'end',
                       gras=(i == 1)))
        p.append(barre(X0, y, val * echelle, 44, couleur, bord))
        p.append(texte(X0 + val * echelle + 14, y + 30,
                       f'{val:.3f} mm', 15, NOIR if i == 1 else GRIS,
                       gras=(i == 1)))

    # fleche de gain
    p.append(f'<line x1="{X0 + phy * echelle}" y1="272" x2="{X0 + sup * echelle}" '
             f'y2="272" stroke="{ACCENT}" stroke-width="2"/>')
    p.append(f'<polygon points="{X0 + phy * echelle},272 {X0 + phy * echelle + 11},266 '
             f'{X0 + phy * echelle + 11},278" fill="{ACCENT}"/>')
    p.append(texte((X0 + phy * echelle + X0 + sup * echelle) / 2, 262,
                   f'{gain:.0%} reduction', 15, ACCENT, 'middle', gras=True))

    p.append(texte(40, 308, f"20 201 targets, one IK branch, 100 epochs. "
                            f"The mixed-branch case and w_phys = 20 are in the README table.",
                   12, GRIS_CLAIR))
    p.append('</svg>')

    out = os.path.join(ASSETS, 'fig_ablation.svg')
    open(out, 'w', encoding='utf-8').write('\n'.join(p))
    print(f"  {out}  ({sup:.3f} -> {phy:.3f} mm, {gain:.0%})")


# =====================================================================
# Figure 2 : le banc d'essai, echelle logarithmique
# =====================================================================
def figure_benchmark():
    import math
    chemin = os.path.join(_ROOT, 'checkpoints', 'benchmark_solveurs.json')
    res = json.load(open(chemin))['resultats']

    libelles = {
        'forme fermee analytique': ('Closed-form analytic', False),
        'reseau, sortie unique': ('Neural, single output', True),
        'reseau, 2 tetes + selection': ('Neural, 2 heads', True),
        'IKPY (repere URDF) *': ('IKPY (iterative) *', False),
        'DLS, depart unique': ('Damped least squares', False),
        'DLS, 8 relances': ('DLS, 8 restarts', False),
    }
    lignes = [(libelles[r['solveur']][0], r['temps_median_ms'], libelles[r['solveur']][1])
              for r in res if r['solveur'] in libelles]
    lignes.sort(key=lambda t: t[1])

    # hauteur : barres + graduations + deux lignes de legende
    W, H = 900, 104 + len(lignes) * 52 + 96
    X0, LARG = 250, 520
    lo, hi = math.log10(0.1), math.log10(1000.0)

    def pos(v):
        return X0 + (math.log10(max(v, 0.1)) - lo) / (hi - lo) * LARG

    p = [entete(W, H)]
    p.append(texte(40, 44, 'Inverse-kinematics solve time', 19, NOIR, gras=True))
    p.append(texte(40, 68, 'Median of 200 calls after 200 warm-up calls, single thread. '
                           'Logarithmic scale.', 13, GRIS))

    base = 104
    for i, (nom, val, neuronal) in enumerate(lignes):
        y = base + i * 52
        p.append(texte(X0 - 18, y + 27, nom, 14, NOIR, 'end', gras=neuronal))
        p.append(barre(X0, y + 8, pos(val) - X0, 32,
                       ACCENT if neuronal else '#d1d5db'))
        p.append(texte(pos(val) + 12, y + 30,
                       f'{val:.3f} ms' if val < 10 else f'{val:.1f} ms',
                       14, NOIR if neuronal else GRIS, gras=neuronal))

    # graduations
    yg = base + len(lignes) * 52 + 4
    for d in (0.1, 1, 10, 100, 1000):
        x = pos(d)
        p.append(f'<line x1="{x}" y1="{base - 8}" x2="{x}" y2="{yg}" '
                 f'stroke="#e5e7eb" stroke-width="1"/>')
        p.append(texte(x, yg + 18, f'{d:g} ms', 11, GRIS_CLAIR, 'middle'))

    p.append(texte(40, yg + 46,
                   'The network is not faster than the closed form. It is faster than generic '
                   'iterative solvers, and it is the only differentiable one.', 12, GRIS))
    p.append(texte(40, yg + 64,
                   '* IKPY is measured in the URDF frame, which does not coincide with this '
                   "project's DH table; only the timing is comparable.", 11, GRIS_CLAIR))
    p.append('</svg>')

    out = os.path.join(ASSETS, 'fig_benchmark.svg')
    open(out, 'w', encoding='utf-8').write('\n'.join(p))
    print(f"  {out}  ({len(lignes)} solveurs)")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    print("Figures generees depuis les fichiers de resultats :")
    figure_ablation()
    figure_benchmark()
