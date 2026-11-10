"""Seat RGB-localized tools with the same Awomo policy and action budget."""
import math

import cv2
import numpy as np

from grounded import EXEMPLARS, GRIP, JOINTS, GroundedGrasp, record

# Layout 0 RGB annotations. Each pair is the corresponding pinch point and
# the landmark toward the head. No simulator asset poses are used.
SLOT_PIXELS = {
    "hammer": ((312, 201), (312, 169)),
    "pliers": ((379, 213), (378, 193)),
    "tape measure": ((370, 175), (386, 175)),
    "wrench": ((340, 210), (353, 207)),
}
TOOL_ORDER = ("tape measure", "hammer", "pliers", "wrench")
HOLD = 8


def turn_between(source, destination):
    return math.degrees(math.atan2(source[0] * destination[1] - source[1] * destination[0],
                                   source @ destination))


def roll_about(axis, degrees):
    x, y = axis / np.linalg.norm(axis)
    normal = np.array([x, y, 0.])
    cross = np.array([[0., 0., y], [0., 0., -x], [-y, x, 0.]])
    cos, sin = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return cos * np.eye(3) + sin * cross + (1 - cos) * np.outer(normal, normal)


def measure_case(world, image):
    cam = world.cameras["cam_head"]
    edges = world.parts.case_edges(image)
    if edges is None:
        return None
    face = world.parts.front_face_height(cam, edges["faces"], world.table)
    if face is None or face["spread_m"] > .012 or not .02 <= face["height_m"] <= .065:
        return None
    return world.parts.find_case(image, cam, face["rim_z"])


def register_slots(world):
    camera = world.cameras["cam_head"]
    reference = measure_case(world, world.reference)
    image = cv2.cvtColor(world.images["cam_head"], cv2.COLOR_RGB2BGR)
    case = measure_case(world, image)
    if reference is None or case is None:
        return None, {}
    axis = world.parts.axis_vector(case["yaw_deg"])
    basis = np.array([axis, [-axis[1], axis[0]]])
    slots = {}
    for name, pixels in SLOT_PIXELS.items():
        # Map the marked surfaces just below the reference rim, then move
        # them with the outline measured from the current head image.
        xy = camera.on_plane(pixels, reference["rim_z"] - .018)[:, :2]
        local = np.stack(world.parts.case_local(reference, xy), axis=-1)
        mapped = local @ basis + case["xy"]
        if world.parts.target_in_case(case, mapped[0]):
            slots[name] = mapped
    return case, slots


def path_chain(world, arm, targets):
    """Plan a sequence without committing any actions."""
    from_tcp = np.eye(4)
    from_tcp[0, 3] = -world.offset
    previous, joints = world.poses[arm], world.state[JOINTS[arm]]
    paths = []
    for index, target in enumerate(targets, start=1):
        try:
            path = world.motion.plan_line(world.planners[arm], world.origins[arm], joints,
                                          previous @ from_tcp, target @ from_tcp)
        except world.motion.PlanFailure as error:
            raise world.motion.PlanFailure(error.reason, f"segment {index}: {error.detail or error.reason}") from error
        paths.append(path)
        previous, joints = target, path[-1]
    return paths


def slow_path(path, start):
    """Use half the joint speed without adding discontinuous hold points."""
    knots = np.vstack((start, path))
    frames = np.arange(len(knots), dtype=float)
    slower = np.arange(1, 2 * len(path) + 1, dtype=float) / 2
    return np.column_stack([np.interp(slower, frames, col) for col in knots.T])


def wrist_evidence(world, arm, name, axis):
    camera = f"cam_{arm}_wrist"
    if camera not in world.cameras:
        return None
    tcp = world.poses[arm][:3, 3]
    unit = axis / np.linalg.norm(axis)
    points = np.stack([tcp + [*(distance * unit), -.006] for distance in (-.028, .025)])
    uv, depth = world.cameras[camera].project(points)
    image = world.images[camera]
    height, width = image.shape[:2]
    pixels = world.parts.classes(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    mask = pixels["yellow" if name in ("pliers", "tape measure") else "part"]
    keep = world.parts.finger_free(mask.shape)
    samples = []
    for (u, v), z in zip(uv, depth, strict=True):
        if z <= .02 or not 11 <= u < width - 11 or not 11 <= v < height - 11:
            continue
        u, v = round(u), round(v)
        patch = np.s_[v-10:v+11, u-10:u+11]
        visible = keep[patch]
        if visible.mean() >= .5:
            samples.append(float(mask[patch][visible].mean()))
    return max(samples, default=None)


def seat_pixels(world, name, point):
    """Count object-colored pixels near a recess landmark in the head view."""
    camera = world.cameras["cam_head"]
    uv, _ = camera.project([[*point, world.case["rim_z"] - .012]])
    u, v = np.rint(uv[0]).astype(int)
    patch = world.images["cam_head"][v-10:v+11, u-10:u+11]
    classes = world.parts.classes(cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    return int(classes["yellow" if name in ("pliers", "tape measure") else "part"].sum())


class SeatedGrasp(GroundedGrasp):
    guard_all_closes = True

    async def run(self, ctx):
        world = ctx.tools["grounded_world"]
        home = world.state.copy()
        record(ctx, "rgb_table", **world.measure_table())
        if world.table is not None:
            world.case, slots = register_slots(world)
            world.slots = slots
            world.initial_seats = {n: seat_pixels(world, n, p[0]) for n, p in slots.items()}
            sources = {n: world.locate(n) for n in EXEMPLARS}
            record(ctx, "rgb_recesses", case=world.case, recesses={n: p.tolist() for n, p in slots.items()})
            for name in TOOL_ORDER:
                if ctx.ended:
                    return
                source = sources[name]
                if name not in slots or source is None:
                    record(ctx, "phase_skipped", tool=name, reason="missing RGB source or recess")
                    continue
                target = self.reconfirm(world, name, source)
                if target is None:
                    record(ctx, "phase_skipped", tool=name, reason="source no longer visible")
                    continue
                self.unsafe = False
                record(ctx, "rgb_tool", tool=name, arm=target.arm, score=target.score,
                       grasp_xy=target.xy.tolist(), head_pixel=target.center.tolist())
                await self.run_phase(ctx, world, name, target)
                if ctx.ended:
                    return
                if self.unsafe:
                    break
                arm = target.arm
                # The transfer has already returned to the source clearance.
                # A straight joint return is used only once the hand is high.
                if world.poses[arm][2, 3] >= world.table + .07:
                    path = world.motion.time_path(np.stack([world.state[JOINTS[arm]], home[JOINTS[arm]]]))
                    await ctx.execute(world.pack_path(arm, path, 1.))
        # Preserve placements and a closed grasp after an unrecoverable
        # refusal. Idle steps still count against the official horizon.
        while not ctx.ended:
            await ctx.execute(np.repeat(world.state[None, :], min(ctx.remaining, 32), axis=0))

    def reconfirm(self, world, name, target):
        # Constrain reacquisition to where this instance was seen before the
        # arms moved. The full RGB view was searched at the episode reset.
        return world.locate(name, near=target.center)

    async def place(self, ctx, world, name, target, pick_plan):
        arm = target.arm
        hold_pose = world.poses[arm].copy()
        hold_joints = world.state[JOINTS[arm]].copy()
        destination, landmark = world.slots[name]
        source_axis = target.head - target.xy
        if name == "tape measure":
            source_axis = target.opening
        dest_axis = landmark - destination
        turn = turn_between(source_axis, dest_axis)
        turns = (turn,)
        if name == "pliers":
            turns = sorted((turn, (turn + 360) % 360 - 180), key=abs)
        if name == "tape measure":
            turn = (turn + 45) % 90 - 45
            turns = (turn, turn + 90, turn - 90)

        wrist_start = wrist_evidence(world, arm, name, source_axis)
        carry = hold_pose.copy()
        refusals = []
        # The closest-to-level solution gets priority. Rolling about the
        # tool's long axis can bring the wrist within reach of the same
        # destination without swinging a long tool across a neighboring slot.
        side = 1 if arm == "left" else -1
        preferred = side * (-1 if dest_axis[1] >= 0 else 1)
        approaches = ((0, .135), (0, .105), (preferred * 12, .110), (-preferred * 12, .110),
                      (preferred * 24, .115), (-preferred * 24, .115),
                      (preferred * 39, .110), (-preferred * 39, .110))
        slow = name != "tape measure"
        for roll, clearance in approaches:
            for yaw in turns:
                rotation = world.parts.rot_z(yaw)
                carry[2, 3] = max(world.table + clearance, world.case["rim_z"] + .058)
                above = carry.copy()
                above[:3, :3] = roll_about(dest_axis, roll) @ rotation @ above[:3, :3]
                above[:2, 3] = destination
                lower = above.copy()
                lower[2, 3] = pick_plan[2] + .007
                try:
                    rise, transfer, descent = path_chain(world, arm, (carry, above, lower))
                except world.motion.PlanFailure as error:
                    refusals.append(f"roll {roll}, yaw {yaw:.0f}: {error.detail or error.reason}")
                    continue
                multiplier = 3 if slow else 2
                if multiplier * sum(map(len, (rise, transfer, descent))) + 2 * HOLD + 36 <= ctx.remaining:
                    break
                refusals.append("insufficient steps for descent and retreat")
            else:
                continue
            break
        else:
            record(ctx, "seat_refused", tool=name, reason="no bounded transfer", refusals=refusals)
            await self.set_down(ctx, world, name, target, hold_pose, pick_plan[2])
            return True

        record(ctx, "seat_plan", tool=name, turn_deg=yaw, roll_deg=roll, target=lower[:3, 3].tolist(),
               wrist_lift_fraction=wrist_start, carry_steps=sum(map(len, (rise, transfer, descent))) * (2 if slow else 1))
        for path in (rise, transfer):
            move = slow_path(path, world.state[JOINTS[arm]]) if slow else path
            await ctx.execute(world.pack_path(arm, move, 0.))
            if ctx.ended:
                return True
            if path is rise:
                record(ctx, "carry_lift", tool=name, finger_joint_m=world.aperture[arm])
                if world.aperture[arm] <= .0025:
                    retreat = np.vstack((rise[-2::-1], hold_joints[None, :]))
                    await ctx.execute(world.pack_path(arm, retreat, 0.))
                    record(ctx, "carry_lost", tool=name, reason="closed fingers before translation")
                    return True
        arrived = world.poses[arm][:3, 3]
        axis = rotation[:2, :2] @ source_axis
        wrist_carry = wrist_evidence(world, arm, name, axis)
        wrist_lost = wrist_start is not None and wrist_start >= .09 and (
            wrist_carry is not None and wrist_carry < .035)
        can_lower = not ctx.ended and world.aperture[arm] > .0025 and not wrist_lost and (
            np.linalg.norm(arrived - above[:3, 3]) <= .022)
        opened = False
        if can_lower:
            move = slow_path(descent, world.state[JOINTS[arm]]) if slow else descent
            await ctx.execute(world.pack_path(arm, move, 0.))
            if ctx.ended:
                return True
            await ctx.execute(world.pack_path(arm, np.repeat(world.state[None, JOINTS[arm]], 3, axis=0), 0.))
            if ctx.ended:
                return True
            tcp = world.poses[arm]
            delta = tcp[:3, 3] - lower[:3, 3]
            angle = float(world.motion.geo.angle_between_deg(tcp[:3, :3], lower[:3, :3]))
            aperture = world.aperture[arm]
            xy_tolerance, height_tolerance = .014, .012
            if name == "tape measure" and wrist_carry is not None and wrist_carry >= .15:
                # The compact body can meet the molded floor higher than a
                # thin shaft. Accept that contact only near the slot center.
                xy_tolerance, height_tolerance = .009, .024
            opened = aperture > .0025 and np.linalg.norm(delta[:2]) <= xy_tolerance and (
                -.012 <= delta[2] <= height_tolerance and tcp[2, 3] <= world.case["rim_z"] + .004 and angle <= 9.)
            record(ctx, "seat_guard", tool=name, tcp_error=delta.tolist(), yaw_error_deg=angle,
                   finger_joint_m=aperture, wrist_carry_fraction=wrist_carry, release=bool(opened))
            if opened:
                await ctx.execute(world.pack_path(arm, np.repeat(world.state[None, JOINTS[arm]], HOLD, axis=0), 1.))
                if ctx.ended:
                    return True
            back_up = np.vstack((descent[-2::-1], transfer[-1:]))
            await ctx.execute(world.pack_path(arm, back_up, float(opened)))
        else:
            record(ctx, "seat_refused", tool=name, reason="carry check failed",
                   tcp=arrived.tolist(), finger_joint_m=world.aperture[arm], wrist_carry_fraction=wrist_carry)
        if ctx.ended:
            return True
        # Retrace the approach above the tray with the fingers in their
        # current commanded state. The retreat clears the head RGB view.
        retreat = np.vstack((transfer[-2::-1], rise[-1:], rise[-2::-1], hold_joints[None, :]))
        await ctx.execute(world.pack_path(arm, retreat, float(opened)))
        if opened:
            increase = seat_pixels(world, name, destination) - world.initial_seats[name]
            record(ctx, "seat_observation", tool=name, object_pixel_increase=increase,
                   rgb_consistent=bool(increase >= 28))
        elif world.aperture[arm] <= .0025:
            record(ctx, "carry_lost", tool=name, reason="closed fingers after retreat")
        elif not ctx.ended:
            await self.set_down(ctx, world, name, target, hold_pose, pick_plan[2])
        return True

    async def set_down(self, ctx, world, name, target, hold_pose, grasp_z):
        arm = target.arm
        above = hold_pose.copy()
        lower = hold_pose.copy()
        lower[2, 3] = grasp_z + .006
        try:
            return_to_source, descent, retreat = path_chain(world, arm, (above, lower, above))
        except world.motion.PlanFailure as error:
            record(ctx, "recovery_refused", tool=name, reason=error.detail or error.reason)
            self.unsafe = True
            return
        if sum(map(len, (return_to_source, descent, retreat))) + HOLD > ctx.remaining:
            record(ctx, "recovery_refused", tool=name, reason="insufficient action budget")
            self.unsafe = True
            return
        for path in (return_to_source, descent):
            await ctx.execute(world.pack_path(arm, path, 0.))
            if ctx.ended:
                return
        delta = world.poses[arm][:3, 3] - lower[:3, 3]
        if np.linalg.norm(delta[:2]) > .018 or abs(delta[2]) > .018:
            record(ctx, "recovery_refused", tool=name, reason="source setdown pose not reached", tcp_error=delta.tolist())
            self.unsafe = True
            return
        await ctx.execute(world.pack_path(arm, np.repeat(world.state[None, JOINTS[arm]], HOLD, axis=0), 1.))
        if ctx.ended:
            return
        await ctx.execute(world.pack_path(arm, retreat, 1.))
        record(ctx, "recovery_setdown", tool=name)
