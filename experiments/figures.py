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
    p.append(texte(40, 44, 'Effect of the physics term on validation error', 19, NOIR, gras=True))
    p.append(texte(40, 68, 'Same data, seed, architecture and schedule. Only the loss differs.',
                   13, GRIS))

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

    p.append(texte(40, 308, f"20,201 targets, one IK branch, 100 epochs. "
                            f"The mixed-branch case and w_phys = 20 are in docs/experiments.md.",
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
                   'The closed form stays 2.3x faster than the network. The network is the only '
                   'differentiable solver in this set.', 12, GRIS))
    p.append(texte(40, yg + 64,
                   '* IKPY is measured in the URDF frame, which does not coincide with this '
                   "project's DH table; only the timing is comparable.", 11, GRIS_CLAIR))
    p.append('</svg>')

    out = os.path.join(ASSETS, 'fig_benchmark.svg')
    open(out, 'w', encoding='utf-8').write('\n'.join(p))
    print(f"  {out}  ({len(lignes)} solveurs)")


# =====================================================================
# Figure 3 : l'architecture, lue dans le checkpoint livre
# =====================================================================
def boite(x, y, w, h, titre, sous_titre='', fond='#f9fafb', bord='#d1d5db',
          couleur=NOIR):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fond}" '
         f'stroke="{bord}" stroke-width="1.5"/>']
    if sous_titre:
        s.append(texte(x + w / 2, y + h / 2 - 3, titre, 14, couleur, 'middle', gras=True))
        s.append(texte(x + w / 2, y + h / 2 + 15, sous_titre, 11, GRIS, 'middle'))
    else:
        s.append(texte(x + w / 2, y + h / 2 + 5, titre, 14, couleur, 'middle', gras=True))
    return '\n'.join(s)


def fleche(x1, y1, x2, y2, couleur=GRIS, pointilles=False):
    d = ' stroke-dasharray="6 4"' if pointilles else ''
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    px, py = x2 - 9 * math.cos(a), y2 - 9 * math.sin(a)
    gx, gy = px - 5 * math.sin(a), py + 5 * math.cos(a)
    hx, hy = px + 5 * math.sin(a), py - 5 * math.cos(a)
    return (f'<line x1="{x1}" y1="{y1}" x2="{px:.1f}" y2="{py:.1f}" stroke="{couleur}" '
            f'stroke-width="1.8"{d}/>'
            f'<polygon points="{x2},{y2} {gx:.1f},{gy:.1f} {hx:.1f},{hy:.1f}" fill="{couleur}"/>')


def figure_architecture():
    import torch
    ckpt = torch.load(os.path.join(_ROOT, 'checkpoints', 'pinn_model_true_physics.pth'),
                      map_location='cpu', weights_only=True)
    h = ckpt['hidden_dim']
    n_params = sum(v.numel() for k, v in ckpt['state_dict'].items()
                   if k.startswith('net.'))

    W, H = 980, 384
    p = [entete(W, H)]
    p.append(texte(40, 40, 'PINN6DOF and its training loss', 19, NOIR, gras=True))
    milliers = f'{n_params:,}'
    p.append(texte(40, 63, f'Read from checkpoints/pinn_model_true_physics.pth: '
                           f'{milliers} parameters, hidden width {h}.', 13, GRIS))

    # chemin d'inference
    y = 110
    p.append(boite(40, y, 110, 56, 'target p*', '(x, y, z) in m'))
    p.append(fleche(150, y + 28, 180, y + 28))
    p.append(boite(180, y, 100, 56, 'normalise', 'mean / std'))
    p.append(fleche(280, y + 28, 310, y + 28))
    for i in range(3):
        x = 310 + i * 120
        p.append(boite(x, y, 104, 56, f'Linear {h}', 'SiLU', fond=ACCENT_FOND,
                       bord=ACCENT, couleur=ACCENT))
        p.append(fleche(x + 104, y + 28, x + 120, y + 28))
    p.append(boite(670, y, 100, 56, 'Linear 6', 'de-normalise', fond=ACCENT_FOND,
                   bord=ACCENT, couleur=ACCENT))
    p.append(fleche(770, y + 28, 800, y + 28))
    p.append(boite(800, y, 140, 56, 'q1 ... q6', 'joint angles, rad'))
    p.append(texte(560, y - 14, 'used at inference: one forward pass, no iteration',
                   12, GRIS_CLAIR, 'middle'))

    # chemin d'entrainement
    y2 = 250
    p.append(fleche(870, y + 56, 870, y2))
    p.append(boite(760, y2, 180, 60, 'differentiable FK', 'DH model in PyTorch'))
    p.append(fleche(760, y2 + 30, 640, y2 + 30))
    p.append(boite(470, y2, 170, 60, 'pose reached', 'position p, rotation R'))
    p.append(fleche(470, y2 + 30, 350, y2 + 30))
    p.append(boite(40, y2 - 10, 310, 80, '', fond='#ffffff', bord=NOIR))
    p.append(texte(56, y2 + 14, 'loss', 14, NOIR, gras=True))
    p.append(texte(56, y2 + 36, "0.1 · ‖q − q_ref‖²   (data term)", 13, GRIS))
    p.append(texte(56, y2 + 56, "+ ‖p − p*‖² + r_tool² · ‖R − R_ref‖"
                                '<tspan baseline-shift="sub" font-size="10">F</tspan>'
                                '<tspan baseline-shift="super" font-size="10">2</tspan>',
                   13, NOIR))
    p.append(fleche(330, y2 - 10, 330, y + 60, ACCENT, pointilles=True))
    p.append(texte(342, y2 - 34, 'backprop through the FK', 12, ACCENT))
    p.append(texte(342, y2 - 18, 'into the weights', 12, ACCENT))

    p.append(texte(500, H - 22, 'r_tool = 0.05 m turns a rotation error into the tip '
                                'displacement it causes. R_ref = FK(q_ref).',
                   11, GRIS_CLAIR, 'middle'))
    p.append('</svg>')

    out = os.path.join(ASSETS, 'fig_architecture.svg')
    open(out, 'w', encoding='utf-8').write('\n'.join(p))
    print(f"  {out}  ({n_params} parametres)")


# =====================================================================
# Figure 4 : nombre de tetes, precision et couverture des branches
# =====================================================================
def figure_multihead():
    import math
    lignes = []
    for k in (1, 2, 4, 8):
        d = json.load(open(os.path.join(_ROOT, 'checkpoints', f'multihypothesis_K{k}.json')))
        lignes.append((k, d['position_mm'], d['rappel_global']))

    W, H = 900, 356
    p = [entete(W, H)]
    p.append(texte(40, 40, 'Multi-head IK on branch-mixed data, no branch labels', 19, NOIR,
                   gras=True))
    p.append(texte(40, 63, 'Same cached dataset, seed and 100 epochs for every K. '
                           'Best-of-K selection by forward kinematics.', 13, GRIS))

    # panneau gauche : erreur de position, echelle log
    X0, Y0, LARG, HAUT = 70, 110, 330, 160
    lo, hi = math.log10(0.1), math.log10(20)
    p.append(texte(X0, Y0 - 14, 'position error (mm, log scale)', 13, NOIR, gras=True))
    for i, (k, err, _) in enumerate(lignes):
        x = X0 + i * (LARG / 4) + 14
        hb = (math.log10(err) - lo) / (hi - lo) * HAUT
        meilleur = err == min(e for _, e, _ in lignes)
        p.append(barre(x, Y0 + HAUT - hb, 52, hb, ACCENT if meilleur else '#d1d5db'))
        p.append(texte(x + 26, Y0 + HAUT - hb - 8, f'{err:.3f}' if err < 1 else f'{err:.2f}',
                       12, NOIR if meilleur else GRIS, 'middle', gras=meilleur))
        p.append(texte(x + 26, Y0 + HAUT + 20, f'K = {k}', 12, GRIS, 'middle'))

    # panneau droit : rappel des branches
    X1 = 500
    p.append(texte(X1, Y0 - 14, 'branches recovered (of those that exist)', 13, NOIR,
                   gras=True))
    for i, (k, _, rap) in enumerate(lignes):
        x = X1 + i * (LARG / 4) + 14
        hb = rap * HAUT
        p.append(barre(x, Y0 + HAUT - hb, 52, max(hb, 1), ACCENT if k == 8 else '#d1d5db'))
        p.append(texte(x + 26, Y0 + HAUT - hb - 8, f'{rap:.0%}', 12,
                       NOIR if k == 8 else GRIS, 'middle', gras=(k == 8)))
        p.append(texte(x + 26, Y0 + HAUT + 20, f'K = {k}', 12, GRIS, 'middle'))

    p.append(texte(40, H - 22, 'Two heads give the best single answer; '
                               'more heads are needed to recover the solution set.',
                   12, GRIS))
    p.append('</svg>')

    out = os.path.join(ASSETS, 'fig_multihead.svg')
    open(out, 'w', encoding='utf-8').write('\n'.join(p))
    print(f"  {out}  (K = 1, 2, 4, 8)")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    print("Figures generees depuis les fichiers de resultats :")
    figure_ablation()
    figure_benchmark()
    figure_architecture()
    figure_multihead()
