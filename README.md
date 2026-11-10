# Awomo Toolbox Harness

A training-free RGB grasp and placement controller around the released Awomo-0.5I policy. It runs the official RoboDojo `store_tools_in_toolbox` task through HarnessStudio. The episode runner retains the native eight-action policy chunk and one policy history frame per simulator action.

## Layout 0 development result

| Controller | Official partial score | Full-task success | Tool credited | Awomo calls |
| --- | ---: | ---: | --- | ---: |
| Native Awomo | 0/100 | 0/1 | None | 113 |
| Camera-guided grasp | 25/100 | 0/1 | Hammer | 99 |
| Camera-guided grasp and placement | 25/100 | 0/1 | Tape measure | 12 |

All three runs used the same initial tool and tray positions, policy seed 0, and **900 controlled actions**. Releases, retreats, and holds count toward that cap. Full success requires all four tools to satisfy the official slot checks and both arms to return home. These are development rollouts from one layout. The final placement run used the local frozen checkpoint and zero external inference calls. See the [run receipts](evidence/layout-0.json).

## Controller

- `native` runs the released Awomo policy on the task instruction.
- `grounded` checks Awomo's proposals for wrong-arm motion and unsafe closes. A camera-guided grasp takes over when necessary, then Awomo handles placement.
- `seated` adds tray registration, bounded transfer planning, and guarded release. It combines a measured wrist aperture, wrist RGB, and fingertip tracking. It also tries a bounded return to the source surface when the release is refused and the tool remains held.

The controller uses RGB, camera calibration, and robot state. The annotated [reference image](reference/layout-0-head.png) comes from layout 0. Recess labels were checked offline against the released task annotations. Simulator object poses are recorded for post-run audit only. The pliers slipped during lift and the hammer and wrench slipped during transfer in the final placement run.

## Run the comparison

Use a Linux NVIDIA machine with an Isaac Lab/RoboDojo simulator environment and a separate Python 3.11 policy environment. The measured runs used separate NVIDIA A6000 GPUs for simulation and inference. These are the source revisions used in the rollouts.

| Dependency | Revision |
| --- | --- |
| [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) | `09bd5f5e762b949f5858ece31347ce62057b3a5e` |
| [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | `6baccd57ff958570d0db1588370289ee36c8bb70` |
| [HarnessStudio](https://github.com/seemandhar/harnessstudio) | `785515ff63cacf70617a1bc884cf8e118ce75b85` |
| [HexaShell](https://github.com/HexaFuture/HexaShell) | `24a5b00e9354147c6e060a8cd1545761073758d2` |

Install RoboDojo with its assets and simulator dependencies using the upstream instructions. Set `DOJO_ROOT`, `STUDIO_ROOT`, and `HEXASHELL_ROOT` to the pinned checkouts. Set `SIM_PYTHON` to the RoboDojo environment's Python. Install HarnessStudio's dependencies and the XPolicyLab client dependencies into that environment. Clone the pinned XPolicyLab revision as `xpolicylab/` inside this checkout.

The benchmark machine used Isaac Lab at `afca7b09d60d8beb9c1cb28b43066499940b969b`, with its Starlette package pin changed from `0.49.1` to `0.45.3` for installation. The policy server used Python 3.11 and PyTorch 2.8.0 with CUDA 12.8.

From the repository root, install the released Awomo policy dependencies, fetch the pinned checkpoints, and start the loopback server. Choose an available GPU index for `POLICY_GPU`.

```bash
python3.11 -m venv .venv-awomo
.venv-awomo/bin/python -m pip install -r xpolicylab/policy/Awomo05/requirements.txt accelerate
.venv-awomo/bin/python fetch_weights.py
ln -s "$DOJO_ROOT/env_cfg" env_cfg
export AWOMO05_WEIGHTS="$PWD/models/awomo"
export AWOMO05_QWEN_VL="$PWD/models/qwen3-vl-4b"
export AWOMO05_FLUX2_VAE="$PWD/models/flux2/vae"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES="$POLICY_GPU" .venv-awomo/bin/python -u xpolicylab/setup_policy_server.py \
  --config_path xpolicylab/policy/Awomo05/deploy.yml \
  --overrides port=23655 host=127.0.0.1 bench_name=RoboDojo \
  task_name=store_tools_in_toolbox env_cfg_type=arx_x5 seed=0 policy_name=Awomo05 \
  action_type=joint input_color_order=bgr ckpt_local_copy=false
```

In another terminal, set the same checkout paths and choose the simulator GPU index in `SIM_GPU`. Use `--smoke-steps 8` for a wiring check. The commands below use the full benchmark horizon.

```bash
for arm in native grounded seated; do
  CUDA_VISIBLE_DEVICES="$SIM_GPU" "$SIM_PYTHON" -u run_toolbox.py \
    --dojo "$DOJO_ROOT" --studio "$STUDIO_ROOT" --hexashell "$HEXASHELL_ROOT" \
    --arm "$arm" --layout 0 --seed 0 --port 23655 --output "runs/layout-0/$arm" \
    --headless --enable_cameras --device cuda:0
done
```

Each run writes `RESULT.json`, the camera MP4s, and `OBJECT_MOTION.json` to its output directory. The model checkpoints and simulator assets retain their upstream licenses.
