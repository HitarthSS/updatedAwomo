"""Drive the official RoboDojo environment through HarnessStudio with native Awomo history."""
import argparse
from dataclasses import asdict
import functools
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
HOME = Path.home()
TASK = "store_tools_in_toolbox"
HORIZON = 900
JOINT_KEYS = ("left_arm_joint_state", "left_ee_joint_state",
              "right_arm_joint_state", "right_ee_joint_state")
SPLITS = (6, 7, 13)
SCENE_OBJECTS = ("toolbox", "hammer", "pliers", "tape_measure", "wrench")


def pack(values):
    import numpy as np

    result = np.concatenate([np.asarray(values[k], dtype=np.float32).reshape(-1) for k in JOINT_KEYS])
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError("Expected fourteen finite ARX-X5 joint values")
    return result


def unpack(action):
    import numpy as np

    return {key: part.tolist() for key, part in zip(JOINT_KEYS, np.split(action, SPLITS), strict=True)}


class HistoryPolicy:
    profile = SimpleNamespace(capabilities=frozenset({"act"}), seeded=False)

    def __init__(self, client):
        self.client = client
        self.pending = None
        self.instruction = None
        self.updates = 0
        self.seconds = 0.0

    def _update(self, instruction):
        self.client.call(func_name="update_obs_batch", obs=[dict(self.pending, instruction=instruction)])
        self.pending = None
        self.updates += 1

    def observe(self, raw):
        # Keep one history frame per physical step. Defer the newest frame
        # until the controller chooses the instruction for its next decision.
        if self.pending is not None:
            self._update(self.instruction)
        self.pending = raw

    def finish(self):
        if self.pending is not None:
            self._update(self.instruction)

    def action_chunk(self, instruction):
        import numpy as np

        if self.pending is None:
            raise RuntimeError("Policy queried twice without a fresh observation")
        self.instruction = instruction
        start = time.monotonic()
        self._update(self.instruction)
        response = self.client.call(func_name="get_action_batch", obs=[0])
        self.seconds += time.monotonic() - start
        if not response or len(response) != 1 or len(response[0]) != 8:
            raise RuntimeError("Awomo did not return its native eight-action chunk")
        return np.stack([pack(action) for action in response[0]])

    async def act(self, public):
        return self.action_chunk(public["task_description"])


class CaptureHistory:
    """Record the upstream inputs during the short native parity rollout."""
    def __init__(self, client):
        self.client = client
        self.frames = []

    def call(self, func_name, obs=None):
        if func_name == "update_obs_batch":
            self.frames.append(obs[0])
        return self.client.call(func_name=func_name, obs=obs)


class HoldPositionSmoke:
    """Exercise the simulator and history transport using a synthetic chunk."""
    def __init__(self, **_):
        self.raw = None

    def call(self, func_name, obs=None):
        if func_name == "reset":
            self.raw = None
        elif func_name == "update_obs_batch":
            self.raw = obs[0]
        elif func_name == "get_action_batch":
            return [[unpack(pack(self.raw["state"])) for _ in range(8)]]
        else:
            raise ValueError(func_name)

    def close(self):
        pass


def create_env(args, app):
    from curobo.types import DeviceCfg
    import env.planner_manager.curobo_planner as planner
    from omegaconf import OmegaConf
    import torch
    from src.eval_client import eval_env
    from task.RoboDojo import task_registry
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    cfg = args.dojo / "env_cfg"
    evaluation = load_yaml(str(cfg / "arx_x5.yml"))
    evaluation.update(task_name=TASK, num_envs=1, device_id=0, eval_batch=True,
                      policy_name="Awomo05", additional_info=args.arm,
                      seed=args.seed, eval_num=1, physx_monitor_enabled=False)
    config = OmegaConf.create({
        **{name: load_yaml(str(cfg / name / (evaluation["config"][name] + ".yml")))
           for name in ("sim", "scene", "camera", "robot")},
        "task_env": load_yaml(task_registry.task_config_path(str(args.dojo / "task/RoboDojo/config"), TASK)),
        "eval_cfg": evaluation,
        "deploy_cfg": {"policy_name": "Awomo05", "host": "127.0.0.1", "port": args.port},
    })
    config, _ = process_config(process_randomization(config), TASK)
    config.sim.scene.num_envs = 1
    config.sim.seed = [0]
    config.camera.default_frequency = 25
    torch.cuda.set_device(torch.device(args.device))
    planner.DeviceCfg = functools.partial(DeviceCfg, device=torch.device(args.device))
    if args.hold_smoke:
        eval_env.WsModelClient = HoldPositionSmoke
    env = eval_env.create_eval_env(config, app)
    env.save_dir = str(args.output)
    env._stream_dir = str(args.output / "_stream")
    return env


def run_episode(args, env):
    import anyio
    from anyio.from_thread import start_blocking_portal
    from harnessstudio.context.bridge import Bridge
    from harnessstudio.context.episode import CaseKey, EpisodeContext
    from harnessstudio.context.trace import fingerprint
    from harnessstudio.contracts import load_contract
    from harnessstudio.core.arms import Arm
    from harnessstudio.core.method import validate
    from methods import method
    from policy.Awomo05.deploy import _get_valid_obs_batch, eval_one_episode_batch

    contract = load_contract("robodojo")
    if contract.horizon({"name": TASK}) != env.step_lim or env.step_lim != HORIZON:
        raise RuntimeError(f"HarnessStudio and RoboDojo horizons differ: {env.step_lim}")
    if args.smoke_steps is not None:
        env.step_lim = args.smoke_steps
    env.reset(seed=[args.layout])
    env.run_reward()
    env.get_score()
    motion = []

    def audit_scene():
        # For post-run diagnosis only. The controller receives RGB and joints.
        layout_manager = env.scene_manager.layout_manager
        centers = {}
        for name in SCENE_OBJECTS:
            position, _ = layout_manager.get_instance_pose(0, label=name)
            if position is not None:
                values = position.tolist()
                centers[name] = values if all(map(math.isfinite, values)) else None
        motion.append({"step": int(env.take_action_cnt[0]), "centers": centers})

    def capture():
        return _get_valid_obs_batch(env, [0])[0]

    def public(raw):
        return dict(task_description=raw["instruction"], state=pack(raw["state"]),
                    images={k: v["color"] for k, v in raw["vision"].items()})

    smoke_actions = []
    protocol_parity = None
    if args.arm == "upstream":
        # Independent deployment-loop parity check, capped for smoke only.
        original = env.take_action_batch
        hashes = []

        def traced(actions, env_idx_list):
            packed = pack(actions[0])
            hashes.append(fingerprint(packed))
            smoke_actions.append(packed.tolist())
            return original(actions, env_idx_list)

        env.take_action_batch = traced
        recorded = CaptureHistory(env.model_client)
        eval_one_episode_batch(env, recorded)
        if not recorded.frames:
            raise RuntimeError("The upstream loop captured no policy frames")
        replay = HistoryPolicy(env.model_client)
        env.model_client.call(func_name="reset")
        first, *remaining = recorded.frames
        replay.observe(first)
        replay_hashes = [fingerprint(a) for a in replay.action_chunk(first["instruction"])]
        for frame in remaining:
            replay.observe(frame)
        replay.finish()
        protocol_parity = dict(action_match=replay_hashes[:len(hashes)] == hashes,
                               history_frames=len(recorded.frames), replay_frames=replay.updates,
                               initial_observation_sha256=fingerprint(public(first)))
        accounting, events = {}, []
    else:
        setup = method(args.arm)
        validate(setup)
        policy = HistoryPolicy(env.model_client)
        env.model_client.call(func_name="reset")
        bridge = Bridge()
        tools = {}
        if args.arm in ("grounded", "seated"):
            from grounded import GroundedWorld

            tools["grounded_world"] = GroundedWorld(env, args.hexashell, args.appearance_reference)
        ctx = EpisodeContext(case=CaseKey("robodojo", TASK, args.layout), contract=contract,
                             horizon=env.step_lim, arm=Arm.CANDIDATE, namespace="awomo-toolbox",
                             policy=policy, budgets=setup.budgets, bridge=bridge,
                             tools=tools,
                             development=args.layout == 0 or args.smoke_steps is not None)

        def set_observation():
            raw = capture()
            policy.observe(raw)
            observation = public(raw)
            if "grounded_world" in tools:
                tools["grounded_world"].capture(observation["images"], observation["state"])
            ctx._set_observation(observation)
            return fingerprint(observation)

        initial = set_observation()
        if args.smoke_steps is None:
            audit_scene()

        async def drive_controller():
            async with anyio.create_task_group() as group:
                bridge.start(group, lambda: setup.controller.build().run(ctx))

        # Isaac owns the main thread. Its USD asset checks can run a nested
        # event loop, so only the HarnessStudio controller is asynchronous.
        with start_blocking_portal() as portal:
            controller_task = portal.start_task_soon(drive_controller)
            try:
                action = portal.call(bridge.first)
                while action is not None:
                    if ctx.ended or ctx.steps >= env.step_lim:
                        raise RuntimeError("Action emitted after the episode limit")
                    env.take_action(unpack(action))
                    ctx._note_sent(action)
                    if args.smoke_steps is not None:
                        smoke_actions.append(action.tolist())
                    if args.smoke_steps is None:
                        audit_scene()
                    if ctx.steps != env.take_action_cnt[0]:
                        raise RuntimeError("HarnessStudio and RoboDojo action counters diverged")
                    if env.end_flag[0]:
                        policy.finish()
                        ctx.stop("success" if env.success[0] else "horizon")
                    else:
                        set_observation()
                    action = portal.call(bridge.next)
            except BaseException:
                portal.call(bridge.cancel)
                raise
            controller_task.result()
        if bridge.error is not None:
            raise bridge.error
        if not env.end_flag[0] or policy.updates != ctx.steps:
            raise RuntimeError(f"Unexpected termination or history length: {policy.updates}/{ctx.steps}")
        ctx.accounting.policy_seconds = policy.seconds
        ctx.accounting.termination = "success" if env.success[0] else "horizon"
        accounting, events, hashes = asdict(ctx.accounting), ctx._events, ctx._action_hashes
        accounting["initial_observation_sha256"] = initial
        accounting["history_frames"] = policy.updates

    success = bool(env.end_flag[0] and env.success[0])
    score = float(env.reward_manager.get_score()[0])
    tool_checks = {}
    if args.arm == "seated" and args.smoke_steps is None:
        # Evaluate the native geometry checks after the controller has stopped.
        # These diagnostics never become controller observations.
        reward = env.reward_manager
        for name in SCENE_OBJECTS[1:]:
            points = ("center", "center_flip") if name == "pliers" else ("center",)
            close, aligned = (
                any(reward.check_once(reward.is_A_functional_point_close_to_B_functional_point(
                    label_A=name, label_B="toolbox", point_A=point, point_B=f"{name}_slot", is_align_qpos=align,
                ), 0) for point in points)
                for align in (False, True)
            )
            tool_checks[name] = dict(close=close, aligned=aligned, covered=reward.check_once(
                reward.is_A_cover_B(label_A="toolbox", label_B=name), 0))
    env.save_video(0, str(args.output / "episode.mp4"), "success" if success else "fail")
    if motion:
        (args.output / "OBJECT_MOTION.json").write_text(json.dumps(motion, allow_nan=False) + "\n")
    return dict(arm=args.arm, seed=args.seed, layout=args.layout, task=TASK, smoke=args.smoke_steps is not None,
                policy="synthetic_hold" if args.hold_smoke else "Awomo-0.5I-Robodojo",
                success=success, partial_score=100 if success else score, steps=env.take_action_cnt[0],
                horizon=env.step_lim, accounting=accounting, events=events, action_hashes=hashes,
                terminal_tool_checks=tool_checks,
                protocol_parity=protocol_parity, smoke_actions=smoke_actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dojo", type=Path, default=HOME / "hexashell-reproduction/RoboDojo")
    parser.add_argument("--studio", type=Path, default=HOME / "guava-harnessstudio-120-20261104/harnessstudio")
    parser.add_argument("--arm", choices=("upstream", "native", "staged", "grounded", "seated"), required=True)
    parser.add_argument("--hexashell", type=Path, default=HOME / "hexashell-reproduction/HexaShell")
    parser.add_argument("--appearance-reference", type=Path, default=ROOT / "reference/layout-0-head.png")
    parser.add_argument("--layout", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=23651)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, choices=(8, 16, 32))
    parser.add_argument("--hold-smoke", action="store_true")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.hold_smoke and args.smoke_steps is None:
        parser.error("--hold-smoke requires a capped smoke run")
    if args.arm == "upstream" and args.smoke_steps != 8:
        parser.error("The upstream parity replay requires one eight-action smoke chunk")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.dojo), str(ROOT / "xpolicylab"), str(args.studio), str(args.hexashell)]
    os.chdir(args.dojo)
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
    add_zero_delay_kit_args(args)
    app = AppLauncher(args).app
    env = None
    try:
        print("Creating RoboDojo environment", flush=True)
        env = create_env(args, app)
        print(f"Starting {args.arm} on official layout {args.layout}", flush=True)
        result = run_episode(args, env)
        temporary = args.output / "RESULT.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "RESULT.json")
    except Exception:
        (args.output / "ERROR.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.model_client.close()
            env.close()
        app.close()


if __name__ == "__main__":
    main()
