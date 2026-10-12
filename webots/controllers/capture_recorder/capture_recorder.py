"""
Records the README demo movie in webots/worlds/capture_pandahand.wbt.

A second, invisible supervisor robot. It starts a 1280x720 movie as soon as the
simulation runs, waits until the red cube has been put down on the tray and the
arm has had time to move away, stops the recording, waits for the file to be
written and quits Webots. The pick-and-place itself is the unchanged
ur5_controller_pandahand controller.

Output: results/webots/capture.mp4 (converted to assets/demo.gif by
experiments/make_demo_gif.py).
"""
import os

from controller import Supervisor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(ROOT, 'results', 'webots', 'capture.mp4')
TRAY_XY = (-1.55, -0.2)          # DEF BLUE_TRAY in the world file
SETTLE_S = 4.0                   # keep filming after the drop, arm moving away
MAX_S = 180.0                    # safety stop, simulated seconds

sup = Supervisor()
dt = int(sup.getBasicTimeStep())
cube = sup.getFromDef('bottle')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

sup.movieStartRecording(OUT, 1280, 720, 0, 90, 1, False)
print(f"[Recorder] recording to {OUT}", flush=True)

lifted, dropped_at = False, None
while sup.step(dt) != -1:
    t = sup.getTime()
    x, y, z = cube.getPosition()
    if z > 1.0:
        lifted = True
    near_tray = abs(x - TRAY_XY[0]) < 0.15 and abs(y - TRAY_XY[1]) < 0.15
    if lifted and dropped_at is None and near_tray and z < 0.96:
        dropped_at = t
        print(f"[Recorder] cube on the tray at t = {t:.1f} s", flush=True)
    if (dropped_at is not None and t - dropped_at > SETTLE_S) or t > MAX_S:
        break

sup.movieStopRecording()
print(f"[Recorder] stopping at t = {sup.getTime():.1f} s", flush=True)
while not sup.movieIsReady():
    if sup.step(dt) == -1:
        break
print(f"[Recorder] movie ready, failed = {sup.movieFailed()}", flush=True)
sup.simulationQuit(0)
