"""Flexiv HIL runtime. Only run_live opens hardware; main defaults to check-only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import select
import signal
import sys
import termios
import time
import tty
import numpy as np
import yaml

from Core import load_config
from Inference.action_processing import canonicalize_action_values, format_policy_state_vector, max_eef_action_delta
from Inference.dispatch import ActionDispatcher, build_policy_state_vector, quaternion_to_rotation6d
from HIL_PI.control import Arbiter, PolicyWorker
from HIL_PI.rtc import rtc_config, make_request
from HIL_PI.execution import preview_candidate, command32
from HIL_PI.tactile_stuck_detection import CombinedStuckDetector
from HIL_PI.stuck_recovery import StuckRecovery
from HIL_PI.keyboard_hil import KeyboardHIL
from HIL_PI.correction import CorrectionBridge, config as correction_config
from HIL_PI.policy_motion import PolicyMotionLimiter
from HIL_PI.recording import Episode, atomic_json
from HIL_PI.spacemouse import SpaceMouse, HumanController, human_action

log = logging.getLogger('hil')
ROOT = Path(__file__).resolve().parents[1]


class Keyboard:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('Live mode requires an interactive terminal for h / Space / p / r / q')
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self):
        # Use the same unbuffered FD for readiness and reads; TextIOWrapper may
        # prefetch additional keys that select() can no longer see.
        if select.select([self.fd], [], [], 0)[0]:
            return os.read(self.fd, 1).decode('ascii', errors='ignore').lower()
        return ''

    def wait(self, choices, message):
        termios.tcflush(self.fd, termios.TCIFLUSH)
        print(message, flush=True)
        while True:
            key = self.read()
            if key in choices and key:
                return key
            time.sleep(.02)

    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def check_config(path):
    cfg = load_config(path)
    raw = yaml.safe_load(Path(path).read_text())
    inf = cfg.inference
    rtc_enabled, _ = rtc_config(raw)
    if not rtc_enabled:
        raise ValueError('This independent PI entrypoint requires inference.rtc.enabled=true')
    if cfg.robot.type != 'flexiv' or inf.action_space.value != 'cartesian':
        raise ValueError('HIL currently supports single-arm Flexiv Cartesian only')
    if inf.policy_format.value != 'normal':
        raise ValueError('PI HIL requires normal 10D wire actions')
    if inf.gripper_mode.value != 'width' or inf.hold_gripper_closed_during_inference:
        raise ValueError('HIL requires width gripper control without forced gripper hold')
    hil = raw.get('hil', {})
    dsrl = hil.get('dsrl_v2', {})
    if dsrl.get('enabled', False):
        if dsrl.get('model_config') != 'flexiv_plug_cord_rtc':
            raise ValueError('DSRL cord collector requires flexiv_plug_cord_rtc')
        if any(hil.get(k, {}).get('enabled', False) for k in
               ('keyboard_hil', 'stuck_detection', 'tactile_stuck_detection',
                'stuck_recovery', 'correction')):
            raise ValueError('DSRL autonomous collection requires recovery/correction disabled')
    if hil.get('server_capture') or hil.get('expo_online'):
        raise ValueError('PI server supports local HIL recording, not server capture or EXPO')
    if type(raw.get('inference', {}).get('gripper_state_is_10x_meters', False)) is not bool:
        raise ValueError('gripper_state_is_10x_meters must be boolean')
    for name in ('linear_speed_m_s', 'angular_speed_rad_s', 'gripper_speed_m_s', 'max_observation_age_s'):
        if not np.isfinite(hil.get(name, 0)) or hil.get(name, 0) <= 0:
            raise ValueError(f'Invalid hil.{name}')
    for name in ('rotation_roll_enabled', 'rotation_pitch_enabled', 'rotation_yaw_enabled'):
        if name in hil and not isinstance(hil[name], bool):
            raise ValueError(f'hil.{name} must be true or false')
    history_frames = hil.get('tactile_history_frames', 30)
    if type(history_frames) is not int or not 1 <= history_frames <= 30:
        raise ValueError('hil.tactile_history_frames must be an integer in [1, 30]')
    home_speed = float(hil.get('home_velocity_scale', .30))
    if not np.isfinite(home_speed) or not 0 < home_speed <= 1:
        raise ValueError('hil.home_velocity_scale must be in (0, 1]')
    if inf.fps <= 0 or inf.n_execute <= 0 or int(hil.get('horizon', 0)) != 50:
        raise ValueError('Expected positive fps/n_execute and horizon=50')
    if len(raw.get('tactile', {})) != 2:
        raise ValueError('Expected two tactile sensors')
    PolicyMotionLimiter(hil.get("policy_motion_limits",{}),inf.fps,10.)
    detector = CombinedStuckDetector(hil.get("stuck_detection", {}), hil.get("tactile_stuck_detection", {}))
    recovery = StuckRecovery(hil.get("stuck_recovery", {}), inf.fps)
    if recovery.enabled and not detector.enabled:
        raise ValueError("stuck_recovery requires stuck_detection.enabled=true")
    correction = correction_config(hil.get("correction", {}))
    if correction["enabled"] and not recovery.enabled:
        raise ValueError("correction requires stuck_recovery.enabled=true")
    if correction["enabled"] and not (ROOT/correction["token_file"]).is_file():
        raise ValueError("correction token_file is missing")
    keyed = KeyboardHIL(hil.get('keyboard_hil'), hil.get('stuck_recovery'), inf.fps)
    if keyed.config['enabled']:
        if detector.enabled or recovery.enabled or correction['enabled']:
            raise ValueError('keyboard_hil requires automatic recovery/correction disabled')
    return cfg, raw


def published_human(dispatcher, requested, gripper_scale):
    return command32(dispatcher, requested, gripper_scale)


def run_live(config_path, cfg, raw, prompt):
    # Delayed imports and explicit --live keep validation entirely hardware-free.
    from Robot import BaseRobot
    from Sensor.manager import SensorManager
    from Sensor.tactile.ring_buffer import TactileRingBuffer
    from HIL_PI.observations import CameraWorker
    hil, inf = raw['hil'], cfg.inference
    dsrl_enabled = bool(hil.get('dsrl_v2', {}).get('enabled', False))
    rtc_enabled, rtc_prefix_length = rtc_config(raw)
    keyed_enabled = bool(hil.get('keyboard_hil', {}).get('enabled', False))
    output = Path(hil.get('output_dir', 'hil_data_pi'))
    if not output.is_absolute():
        output = ROOT/output
    mouse = SpaceMouse(hil.get('deadzone', .1), hil.get('vendor_id', 0x256F), hil.get('product_id'),
                       hil.get('rotation_roll_enabled', True), hil.get('rotation_pitch_enabled', True),
                       hil.get('rotation_yaw_enabled', True))
    robot = gripper = sensors = cameras = worker = episode = correction = None
    rings = []
    smooth = raw.get('inference', {}).get('smooth', {})
    arb = Arbiter(smooth_steps=smooth.get('overlap_steps', 8) if smooth.get('enabled', False) else 0,
                  rtc_enabled=rtc_enabled)
    log.info('HIL RTC enabled=%s prefix_length=%d; smoothing=%s',
             rtc_enabled, rtc_prefix_length, 'full_remaining' if rtc_enabled and arb.smooth_steps else arb.smooth_steps)
    human_controller = HumanController()
    old_signals = {}

    def interrupt(signum, frame):
        # No implicit homing on interrupt, error, or exit.
        if robot is not None and robot.is_connected():
            robot.stop()
        raise KeyboardInterrupt

    def home():
        robot.stop()
        if inf.inference_home is not None:
            ok = robot.move_joint_position(inf.inference_home, velocity=hil.get('home_velocity_scale', .30), timeout_s=60.)
        else:
            ok = robot.go_home(velocity=hil.get('home_velocity_scale', .30))
        if not ok:
            raise RuntimeError('Home did not complete; no inference restart')
        if inf.inference_home_gripper_width is not None:
            gripper.move(float(inf.inference_home_gripper_width), wait=True)
        dispatcher.reset_velocity_tracking()

    try:
        mouse.open()
        # Tactile devices are exclusively owned by TactileRingBuffer below.
        # enabled_cameras may include tactile names for observation routing.
        rgb_names = [name for name in (inf.enabled_cameras if inf.enabled_cameras is not None
                                      else list(cfg.cameras)) if name not in cfg.tactile]
        sensors = SensorManager.from_config(cfg, strict=True, enabled_names=rgb_names)
        sensors.open_all()
        cameras = CameraWorker(sensors, cfg)
        cameras.start()
        for name, spec in raw['tactile'].items():
            ring = TactileRingBuffer(name, spec['mapped_key'], spec['device_path'],
                                     fps=spec.get('fps', 30), width=spec.get('width', 640), height=spec.get('height', 480))
            rings.append(ring)
            ring.start()
            if not ring.wait_for_frames(1):
                raise RuntimeError(f'Tactile not ready: {name}')
        robot = BaseRobot.from_config(config_path)
        robot.connect()
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_signals[sig] = signal.signal(sig, interrupt)
        if robot.is_fault():
            raise RuntimeError('Robot fault; inspect on teach pendant before starting HIL')
        robot.enable()
        if not robot.wait_until_operational(timeout_s=30.):
            raise RuntimeError('Robot not operational')
        gripper = robot.create_gripper()
        if gripper is None:
            raise RuntimeError('Flexiv gripper required')
        gripper.connect()
        # Keep the copied HIL execution/recording layout unchanged internally.
        # Only the PI network boundary converts to/from physical 10D.
        dispatch_cfg = cfg.model_copy(deep=True)
        dispatch_cfg.inference.policy_format = type(inf.policy_format)('canonical')
        dispatch_cfg.inference.canonical_dim = 32
        dispatcher = ActionDispatcher.from_config(robot, gripper, dispatch_cfg)
        if dsrl_enabled:
            from HIL_PI.dsrl_v2_client import DSRLV2Worker
            worker = DSRLV2Worker(inf.server, inf.recv_timeout_ms)
        else:
            worker = PolicyWorker(inf.server, inf.recv_timeout_ms)
        ccfg = correction_config(hil.get("correction", {}))
        if ccfg["enabled"]:
            correction = CorrectionBridge(ccfg, ROOT)
        scale = 10. if inf.gripper_action_is_10x_meters else 1.
        metadata = dict(config=raw, config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                        prompt=prompt, fps=inf.fps, horizon=hil['horizon'],
                        state_layout='xyz_m,rot6d_columns,gripper_m,zero_pad_to_32',
                        action_layout=f'absolute_xyz_m,rot6d_columns,gripper_m_times_{scale},zero_pad_to_32',
                        policy_source='pi_rtc_physical_10d_padded_for_hil',
                        wire_state_dim=10,
                        wire_gripper_state_scale=10. if raw['inference'].get(
                            'gripper_state_is_10x_meters', False) else 1.,
                        sensor_timing='latest asynchronous images; read-complete timestamps, not hardware synchronization',
                        pairing='execution_tick_shadow_policy_v2',
                        hil_variant='keyboard_two_press_hil_v1' if keyed_enabled else 'spacemouse_takeover')
        with Keyboard() as keyboard:
            if keyboard.wait('\r\nq', '硬件已连接；Enter 回起始位置，q 退出。') == 'q':
                return
            home()
            while True:
                if keyboard.wait('\r\nq', 'Enter 开始新一轮推理，q 退出。') == 'q':
                    break
                worker.reset()
                for ring in rings:
                    ring.clear()
                for ring in rings:
                    if not ring.wait_for_frames(1):
                        raise RuntimeError('No tactile frame after reset')
                episode = Episode(output, metadata)
                if dsrl_enabled:
                    worker.begin_dsrl_episode(episode.episode_id, prompt,
                                              hil['dsrl_v2']['model_config'])
                if hil.get('server_capture', False):
                    session = worker.begin_episode(episode.episode_id, metadata)
                    atomic_json(episode.path/'server_transaction.json', dict(session,server=inf.server,sync_status='pending'))
                arb.switch('policy')
                policy_limiter=PolicyMotionLimiter(hil.get('policy_motion_limits',{}),inf.fps,scale)
                stuck_detector = CombinedStuckDetector(hil.get("stuck_detection", {}), hil.get("tactile_stuck_detection", {}))
                stuck_detector.start()
                recovery = StuckRecovery(hil.get("stuck_recovery", {}), inf.fps)
                keyed = KeyboardHIL(hil.get('keyboard_hil'), hil.get('stuck_recovery'), inf.fps)
                correction_motion = StuckRecovery(dict(recovery.config,
                    max_distance_m=ccfg["max_translation_m"], max_rotation_rad=ccfg["max_rotation_rad"],
                    motion_mode=ccfg["motion_mode"],linear_speed_m_s=ccfg["linear_speed_m_s"],
                    angular_speed_rad_s=ccfg["angular_speed_rad_s"]),inf.fps)
                if correction is not None:
                    correction.new_episode()
                correction_captured_start = None
                stuck_log_at = 0.0
                stuck_reason_counts = {}
                if stuck_detector.enabled:
                    log.info("Stuck detection enabled; recovery=%s: %s", recovery.enabled, stuck_detector.config)
                human_controller.reset()
                mouse.activity.clear()
                tick = 0
                next_tick = time.monotonic()
                latency_steps = 5
                reset_requested = False
                resume_request_generation = None
                resume_not_before = 0.0
                last_h_input_time = None
                last_b_input_time = None
                if keyed_enabled:
                    log.info('POLICY：第一次 h 回退，回退完成后第二次 h 进入 HUMAN；空格或 p 恢复；r 重置；q 退出')
                else:
                    log.info('POLICY：移动 SpaceMouse/按夹爪键接管；人工模式按 p 恢复；b 手动回退（需 YAML 启用）；r 重置；q 停止退出')
                while True:
                    resume_human_correction = False
                    resume_after_correction = False
                    correction_started_this_tick = False
                    manual_started_this_tick = False
                    now = time.monotonic()
                    key = keyboard.read()
                    fresh_h = key == 'h' and (last_h_input_time is None or now - last_h_input_time > .6)
                    if key == 'h':
                        last_h_input_time = now
                    if key in ('r', 'q'):
                        reset_requested = key == 'r'
                        if correction is not None:
                            correction.clear()
                        capture_end_time = time.monotonic()
                        if dsrl_enabled and arb.mode == 'policy':
                            robot.stop()
                            terminal_time = time.monotonic()
                            terminal_arm, terminal_grip = robot.observe(), gripper.observe()
                            terminal_state = format_policy_state_vector(build_policy_state_vector(
                                terminal_arm, action_space='cartesian', gripper_width=terminal_grip.width,
                                gripper_max_width=terminal_grip.max_width, gripper_mode='width'),
                                action_space='cartesian', policy_format='canonical', canonical_dim=32)
                            terminal_payload, _ = cameras.snapshot(
                                terminal_state, rings, prompt, hil['max_observation_age_s'],
                                history_frames=hil.get('tactile_history_frames', 1))
                            from HIL_PI.dsrl_decisions import snapshot
                            episode.add('dsrl_terminal', time=terminal_time,
                                        decision=snapshot(arb, terminal_state,
                                                          arb.rtc_prefix(rtc_prefix_length), terminal_time),
                                        observation=terminal_payload)
                        arb.switch('paused')
                        robot.stop()
                        episode.add('episode_end', reason=key, time=capture_end_time,
                                    last_tick=tick-1, homing_recorded=False)
                        episode.metadata.update(capture_end_time=capture_end_time,
                                                capture_end_reason=key, homing_recorded=False)
                        break
                    axes, buttons = mouse.sample()
                    if not keyed_enabled and mouse.activity.is_set() and arb.mode in ('policy', 'recovery', 'recovery_hold', 'correction'):
                        arb.switch('human')
                        log.info('HUMAN：policy 仅作为影子候选继续调度；松开保持，p 交回控制')
                    resume_key = key == 'p' or (keyed_enabled and key == ' ')
                    if resume_key:
                        log.info('[RESUME] %r received mode=%s worker_busy=%s generation=%s',
                                 key, arb.mode, worker.busy, arb.generation)
                        episode.add('policy_resume', tick=tick, event='key_received', mode=arb.mode,
                                    key=key, worker_busy=worker.busy, generation=arb.generation)
                        if arb.mode not in ('human', 'recovery_hold') or (keyed_enabled and arb.mode != 'human'):
                            log.info('[RESUME] key ignored in mode=%s; wait for hold or release human control', arb.mode)
                    if resume_key and arb.mode in ('human', 'recovery_hold') and (not keyed_enabled or arb.mode == 'human'):
                        if np.any(axes) or buttons:
                            log.info('先松开 SpaceMouse，再按%s', '空格或 p' if keyed_enabled else 'p')
                        else:
                            if keyed_enabled:
                                episode.add('hil_transition', event='policy_resumed', tick=tick,
                                            time=time.monotonic(), hil_id=keyed.hil_id, key=key)
                                keyed.reset()
                            resume_human_correction = arb.mode == 'human'
                            if correction is not None and not resume_human_correction:
                                correction.clear()
                            arb.switch('policy')
                            # Keep the actual last command, not measured feedback or
                            # a human-mode shadow prediction, until fresh actions arrive.
                            arb.track(command32(dispatcher, None, scale))
                            policy_limiter.reset()
                            recovery.reset()
                            stuck_detector.reset()
                            human_controller.reset()
                            dispatcher.reset_velocity_tracking()
                            mouse.activity.clear()
                            stuck_detector.calibrate_hold(grip.width)
                            resume_request_generation = arb.generation
                            resume_time = time.monotonic()
                            resume_not_before = resume_time + 0.1
                            episode.add('policy_resume', tick=tick, event='accepted',
                                        time=resume_time, not_before=resume_not_before,
                                        generation=arb.generation)
                            log.info('[RESUME] accepted generation=%s; wait 0.1s before fresh observation/request; %s', arb.generation,
                                     'waiting for in-flight response before fresh request' if worker.busy
                                     else 'worker idle')
                    response = worker.poll()
                    if response is not None:
                        episode.add('prediction', **response)
                        latency_steps = max(1, int(np.ceil((response['response_time']-response['request_time'])*inf.fps)))
                        skipped = max(0, arb.policy_steps-response['policy_steps'])
                        arb.accept(response['response'], response['generation'], inf.n_execute, skipped,
                                   source={k: response[k] for k in ('request_id', 'tick', 'request_time',
                                                                    'observation_time', 'response_time')})
                    observation_generation = arb.generation
                    rtc_prefix = arb.rtc_prefix(rtc_prefix_length)
                    observation_time = time.monotonic()
                    arm = robot.observe()
                    grip = gripper.observe()
                    if len(arm.eef_pose) != 7:
                        raise RuntimeError('Expected a single 7D Flexiv TCP pose')
                    state = format_policy_state_vector(build_policy_state_vector(
                        arm, action_space='cartesian', gripper_width=grip.width,
                        gripper_max_width=grip.max_width, gripper_mode='width'),
                        action_space='cartesian', policy_format='canonical', canonical_dim=32)
                    if not np.isfinite(state).all():
                        raise RuntimeError('Nonfinite robot state')
                    if keyed_enabled:
                        keyed.feedback(observation_time, state)
                    payload, sensor_times = cameras.snapshot(state, rings, prompt, hil['max_observation_age_s'],
                                                              history_frames=hil.get('tactile_history_frames', 30))
                    if rtc_enabled:
                        payload = make_request(
                            payload, rtc_prefix,
                            gripper_state_is_10x_meters=raw['inference'].get(
                                'gripper_state_is_10x_meters', False))
                    if keyed_enabled and fresh_h:
                        if arb.mode == 'policy' and keyed.phase == 'none':
                            prior_width = dispatcher._last_gripper_width_cmd
                            current_width = grip.width if prior_width is None else prior_width
                            last_command = list(state)
                            last_command[9] = current_width * scale
                            result = keyed.trigger(observation_time, state, last_command, gripper_scale=scale)
                            arb.switch('recovery' if keyed.phase in ('opening', 'retreating') else 'recovery_hold')
                            policy_limiter.reset()
                            episode.add('hil_transition', event='rollback_requested', tick=tick,
                                        time=observation_time, **result)
                            if keyed.phase == 'await_human':
                                finished = keyed.reason == 'already_at_target'
                                episode.add('hil_transition', event='rollback_finished' if finished else 'rollback_failed', tick=tick,
                                            time=observation_time, hil_id=keyed.hil_id,
                                            branch=keyed.branch, reason=keyed.reason)
                            log.info('[HIL] h rollback: branch=%s phase=%s reason=%s',
                                     keyed.branch, keyed.phase, keyed.reason)
                        elif arb.mode == 'recovery_hold' and keyed.request_human():
                            episode.add('hil_transition', event='human_requested', tick=tick,
                                        time=observation_time, hil_id=keyed.hil_id)
                    if keyed_enabled and keyed.neutral(*mouse.sample()):
                        arb.switch('human')
                        human_controller.reset()
                        mouse.activity.clear()
                        episode.add('hil_transition', event='human_started', tick=tick,
                                    time=observation_time, hil_id=keyed.hil_id)
                        log.info('[HIL] HUMAN started: hil_id=%s', keyed.hil_id)
                    # Re-check just before dispatch; the control thread is the sole robot writer.
                    if not keyed_enabled and mouse.activity.is_set() and arb.mode in ('policy', 'recovery', 'recovery_hold', 'correction'):
                        arb.switch('human')
                    axes, buttons = mouse.sample()
                    if correction is not None:
                        if arb.mode == 'human':
                            # A failed/interrupted retreat still permits a supervised manual
                            # correction, anchored BEFORE the first human action is dispatched.
                            if correction.anchor is None and recovery.status != 'idle':
                                anchor_command = list(state)
                                width = dispatcher._last_gripper_width_cmd
                                anchor_command[9] = (grip.width if width is None else width) * scale
                                cycle = correction.capture(state, payload, observation_time, sensor_times,
                                                           anchor_command, episode.episode_id, tick)
                                if cycle is not None:
                                    correction.anchor.update(anchor_source='human_takeover_after_recovery',
                                                             recovery_reason=recovery.reason,
                                                             recovery_status=recovery.status)
                                    episode.add('correction_anchor', tick=tick, time=observation_time,
                                                sample_id=cycle, anchor_pose=list(state[:9]),
                                                wrist_image=payload[ccfg['wrist_key']], sensor_times=sensor_times,
                                                source='human_takeover_after_recovery',
                                                recovery_status=recovery.status, recovery_reason=recovery.reason)
                                    log.warning('[CORRECTION] manual anchor after recovery %s: id=%s',
                                                recovery.reason, cycle)
                            correction.human()
                            if correction_motion.status != 'idle':
                                correction_motion.reset()
                                episode.add('correction_execution', tick=tick, time=observation_time, event='human_cancelled')
                        if resume_human_correction:
                            sample = correction.resume(state, payload, observation_time, sensor_times, tick)
                            if sample is not None:
                                episode.add('correction_supervision', tick=tick, time=observation_time, **sample)
                        result = correction.poll(state)
                        if result is not None and arb.mode == 'recovery_hold':
                            target = np.array(state, dtype=float)
                            target[:9] = result['poses'][result['selected']]
                            target[9] = correction.anchor['gripper_command']
                            correction_motion.reset()
                            correction_motion.checkpoint = (observation_time, target)
                            report = correction_motion.start(observation_time, state, target)
                            if report['reason'] in ('stuck_triggered', 'already_at_target'):
                                correction_started_this_tick = True
                                arb.switch('correction')
                                policy_limiter.reset()
                                stuck_detector.reset()
                                episode.add('correction_prediction', tick=tick, time=observation_time, **result)
                                log.warning('[CORRECTION] executing selected=%d score=%.4f model_step=%d',
                                            result['selected'], result['scores'][result['selected']], result['steps'])
                            else:
                                log.warning('[CORRECTION] execution rejected: %s', report['reason'])
                    # Manual rollback precedes policy pop/dispatch; human takeover wins.
                    if key == 'b':
                        fresh_b = last_b_input_time is None or now-last_b_input_time > .6
                        last_b_input_time = now
                        allowed = (fresh_b and hil.get('manual_recovery_enabled', False)
                                   and not keyed_enabled and recovery.enabled and arb.mode == 'policy'
                                   and not mouse.activity.is_set() and not np.any(axes) and not buttons)
                        if allowed:
                            if correction is not None:
                                correction.clear()
                            last_command = list(state)
                            width = dispatcher._last_gripper_width_cmd
                            last_command[9] = (grip.width if width is None else width) * scale
                            report = recovery.start(observation_time, state, last_command)
                            manual_started_this_tick = True
                            arb.switch('recovery' if recovery.status == 'retreating' else 'recovery_hold')
                            stuck_detector.reset()
                            policy_limiter.reset()
                            episode.add('stuck_recovery', tick=tick, time=observation_time,
                                        event='started', trigger='manual_b', **report)
                            log.warning('[RECOVERY manual b] %s: %s target_xyz=%s', recovery.status,
                                        recovery.reason, None if recovery.target is None else recovery.target[:3].tolist())
                        else:
                            log.info('[RECOVERY manual b] ignored mode=%s enabled=%s fresh=%s',
                                     arb.mode, hil.get('manual_recovery_enabled', False), fresh_b)

                    # Snapshot the observation's queue before consuming this tick's action.
                    dsrl_decision = None
                    if dsrl_enabled and arb.mode == 'policy':
                        from HIL_PI.dsrl_decisions import snapshot
                        dsrl_decision = snapshot(arb, state, rtc_prefix, observation_time)
                    # Pop one policy candidate on EVERY tick, even during human control.
                    # It remains a shadow: only the selected action reaches live hardware.
                    schedule_start = arb.policy_steps
                    candidate = arb.actions.popleft() if arb.actions else None
                    arb.advance(candidate)
                    policy_preview = None
                    if candidate is not None:
                        policy_limited = policy_limiter.preview(candidate['action'],state)
                        policy_preview = preview_candidate(dispatcher, policy_limited, arm, inf,
                                                           grip.max_width, scale)
                    # Final input gate before publish: takeover cannot publish the popped policy candidate.
                    if not keyed_enabled and mouse.activity.is_set() and arb.mode in ('policy', 'recovery', 'recovery_hold', 'correction'):
                        arb.switch('human')
                    if correction is not None and arb.mode == 'human':
                        correction.human()
                        correction_motion.reset()
                    human_requested = None
                    recovery_fields = {}
                    if arb.mode == 'human' and recovery.status != 'idle':
                        episode.add('stuck_recovery', tick=tick, time=observation_time,
                                    event='cancelled', reason='human_takeover')
                        log.warning('[RECOVERY] cancelled: human takeover')
                        recovery.reset()
                        stuck_detector.reset()
                        human_controller.reset()
                        policy_limiter.reset()
                    if arb.mode == 'human':
                        human_requested = human_controller.step(state, axes, buttons, 1./inf.fps,
                            linear_speed=hil['linear_speed_m_s'], angular_speed=hil['angular_speed_rad_s'],
                            gripper_speed=hil['gripper_speed_m_s'], gripper_max=grip.max_width, gripper_scale=scale)
                        if arb.policy_steps == schedule_start:
                            arb.policy_steps += 1
                        requested = human_requested
                    elif arb.mode == 'correction':
                        if correction_started_this_tick:
                            requested, correction_report = correction_motion.command.tolist(), correction_motion.report()
                        else:
                            requested, correction_report = correction_motion.step(observation_time, state)
                        recovery_fields['correction_execution'] = correction_report
                        if correction_motion.status == 'holding':
                            completed = correction_motion.reason in ('completed', 'already_at_target')
                            episode.add('correction_execution', tick=tick, time=observation_time,
                                        event='completed' if completed else 'failed', **correction_report)
                            log.warning('[CORRECTION] %s', correction_motion.reason)
                            if completed:
                                resume_after_correction = True
                            else:
                                arb.switch('recovery_hold')
                                recovery.command = np.array(requested, dtype=float)
                    elif keyed_enabled and keyed.active:
                        requested, keyed_result = keyed.step(observation_time, state, grip.width)
                        if keyed_result is not None:
                            if arb.mode == 'recovery':
                                arb.switch('recovery_hold')
                            episode.add('hil_transition', event=keyed_result, tick=tick,
                                        time=observation_time, hil_id=keyed.hil_id,
                                        branch=keyed.branch, reason=keyed.reason)
                            log.info('[HIL] %s: %s; press h to start HUMAN', keyed_result, keyed.reason)
                    elif arb.mode in ('recovery', 'recovery_hold'):
                        previous_status = recovery.status
                        if manual_started_this_tick:
                            requested, recovery_report = recovery.command.tolist(), recovery.report()
                        else:
                            requested, recovery_report = recovery.step(observation_time, state)
                        recovery_fields['stuck_recovery'] = recovery_report
                        if recovery.status == 'holding' and arb.mode == 'recovery':
                            arb.switch('recovery_hold')
                        if previous_status != recovery.status:
                            episode.add('stuck_recovery', tick=tick, time=observation_time,
                                        event='holding', **recovery_report)
                            log.warning('[RECOVERY] holding: %s elapsed_s=%.3f error_mm=%s; p resumes policy, SpaceMouse takes over',
                                        recovery.reason, recovery_report['elapsed_s'],
                                        None if recovery_report['position_error_m'] is None else round(1000*recovery_report['position_error_m'], 3))
                        if (correction is not None and recovery.status == 'holding'
                            and recovery.reason in ('completed', 'already_at_target')
                            and correction_captured_start != recovery.started):
                            correction_captured_start = recovery.started
                            cycle = correction.capture(state, payload, observation_time, sensor_times,
                                                       requested, episode.episode_id, tick)
                            if cycle is not None:
                                episode.add('correction_anchor', tick=tick, time=observation_time,
                                            sample_id=cycle, anchor_pose=list(state[:9]),
                                            wrist_image=payload[ccfg['wrist_key']], sensor_times=sensor_times)
                                correction.predict()
                    elif candidate is not None:
                        if not policy_preview['valid']:
                            raise RuntimeError(policy_preview['rejection'])
                        requested = policy_limited
                    else:
                        # Hold the last command when a response is late; sending measured
                        # pose instead would retract a still-in-progress target.
                        requested = (arb.last_action if arb.last_action is not None else
                            human_action(state, np.zeros(6), 0, 1./inf.fps,
                                         gripper_scale=scale, gripper_max=grip.max_width))
                    if arb.mode == 'policy':
                        requested=policy_limiter.preview(requested,state)
                    action = canonicalize_action_values(requested, 'cartesian', policy_format='canonical',
                                                         canonical_dim=32, effective_action_dim=10)
                    pos, rot = max_eef_action_delta(arm.eef_pose, action)
                    if not np.isfinite(action).all() or pos > inf.max_eef_delta_m or rot > inf.max_eef_rotation_delta_rad:
                        raise RuntimeError(f'Action safety rejected: {pos:.4f}m/{rot:.4f}rad')
                    publish_start = time.monotonic()
                    previous_width_m = dispatcher._last_gripper_width_cmd
                    if previous_width_m is None:
                        previous_width_m = grip.width
                    dispatcher.dispatch(action, state=arm)
                    published_time = time.monotonic()
                    published = command32(dispatcher, requested, scale)
                    if keyed_enabled and arb.mode == 'policy':
                        keyed.executed(observation_time, state, published, candidate,
                                       arb.generation, previous_width_m, scale)
                    policy_limiter.commit(published)
                    stuck_fields = dict(recovery_fields)
                    start_recovery = False
                    if stuck_detector.enabled:
                        raw_frame = raw_stamp = None
                        if stuck_detector.tactile.enabled:
                            for ring in rings:
                                if ring.wire_key == stuck_detector.tactile.config['image_key']:
                                    raw_frame, raw_stamp = ring.latest_bgr_timed()
                                    break
                        stuck_detector.set_observation(payload, sensor_times, raw_frame, raw_stamp, dispatcher._last_gripper_width_cmd)
                        detection = stuck_detector.update(
                            observation_time, state[:3], published[:3],
                            mode=arb.mode, fresh_command=(candidate is not None and arb.mode == 'policy'),
                            source=candidate, generation=arb.generation,
                            chunk_end=(candidate is not None and not arb.actions))
                        stuck_fields['stuck_detection'] = detection
                        recovery.observe(observation_time, state, detection, arb.mode,
                                         source=candidate, generation=arb.generation)
                        start_recovery = recovery.enabled and arb.mode == 'policy' and detection['event'] == 'entered'
                        reason = detection['reason']
                        stuck_reason_counts[reason] = stuck_reason_counts.get(reason, 0) + 1
                        if observation_time >= stuck_log_at:
                            log.info('[STUCK monitor] stuck=%s reason=%s error_mm=%s excursion_mm=%s window_s=%.3f confirm_s=%.3f fresh=%s reasons=%s',
                                     detection['stuck'], reason,
                                     None if detection['tracking_error_m'] is None else round(1000*detection['tracking_error_m'], 3),
                                     None if detection['position_excursion_m'] is None else round(1000*detection['position_excursion_m'], 3),
                                     detection['window_span_s'], detection['suspect_duration_s'],
                                     candidate is not None and arb.mode == 'policy', stuck_reason_counts)
                            if stuck_detector.tactile.enabled:
                                tactile_log = detection.get('tactile') or {}
                                if stuck_detector.tactile.config['method'] == 'monitor':
                                    log.info('[MONITOR] reason=%s critical=%s problem=%s confirm=%s age_ms=%s processing_ms=%s',
                                             tactile_log.get('reason'), tactile_log.get('critical_probability'),
                                             tactile_log.get('problem_probability'), tactile_log.get('confirm_count'),
                                             tactile_log.get('result_age_ms'), tactile_log.get('processing_ms'))
                                slip_log = tactile_log.get('slip') or {}
                                log.info('[TACTILE monitor] stuck=%s reason=%s ready=%s exceeded_blocks=%s slip=%s slip_speed_px_s=%s',
                                         tactile_log.get('stuck'), tactile_log.get('reason'),
                                         tactile_log.get('ready'), tactile_log.get('exceeded_blocks'),
                                         slip_log.get('slipping'), slip_log.get('speed_px_s'))
                            stuck_log_at = observation_time + stuck_detector.config['log_interval_s']
                            stuck_reason_counts = {}

                        if detection['event'] is not None:
                            episode.add('stuck_detection', tick=tick, time=observation_time, **detection)
                            log.warning('[STUCK] %s reason=%s error_m=%s excursion_m=%s recovery_enabled=%s evaluated_chunk=%s',
                                        detection['event'], detection['reason'],
                                        detection['tracking_error_m'], detection['position_excursion_m'], recovery.enabled, detection.get('evaluated_chunk'))

                    if candidate is not None and policy_preview and policy_preview['valid']:
                        arb.track(policy_preview['action'], {k: candidate[k] for k in
                            ('request_id','chunk_index','observation_time') if k in candidate})
                    elif arb.mode == 'policy':
                        arb.track(published)
                    human = published if arb.mode == 'human' else None
                    request_id = None
                    prefetch_steps = min(inf.n_execute, latency_steps + 2)
                    if (not start_recovery and arb.mode in ('policy', 'human') and not worker.busy
                            and (resume_request_generation != arb.generation or observation_time >= resume_not_before)
                            and (not rtc_enabled or observation_generation == arb.generation)
                            and (rtc_enabled or len(arb.actions) <= prefetch_steps)):
                        if dsrl_enabled:
                            if dsrl_decision is None or arb.mode != 'policy':
                                raise RuntimeError('DSRL episode entered non-policy control; discard this episode')
                            payload = dict(payload, extra=dict(payload.get('extra', {}),
                                           dsrl_decision=dsrl_decision))
                        request_id = worker.submit(payload, tick=tick, generation=arb.generation,
                                                   policy_steps=schedule_start, observation_time=observation_time)
                        if dsrl_enabled:
                            episode.add('dsrl_decision', tick=tick, request_id=request_id,
                                        decision=dsrl_decision)
                        if resume_request_generation == arb.generation:
                            log.info('[RESUME] fresh request submitted id=%s tick=%s cmd=%s generation=%s',
                                     request_id, tick, payload.get('cmd', 'predict'), arb.generation)
                            episode.add('policy_resume', tick=tick, event='request_submitted',
                                        request_id=request_id, generation=arb.generation)
                            resume_request_generation = None
                    if resume_request_generation is not None and resume_request_generation != arb.generation:
                        log.info('[RESUME] superseded by mode=%s generation=%s; old resume request discarded',
                                 arb.mode, arb.generation)
                        resume_request_generation = None
                    policy_source = None if candidate is None else {k: v for k,v in candidate.items() if k not in ('action', 'raw_action')}
                    pair_valid = bool(arb.mode == 'human' and policy_preview and policy_preview['valid'])
                    episode.add('tick', tick=tick, time=now, publish_start=publish_start, publish_time=published_time,
                                robot_send_start=dispatcher.last_robot_send_start,
                                robot_send_end=dispatcher.last_robot_send_end,
                                observation_time=observation_time, state_time=arm.timestamp, mode=arb.mode, segment=arb.segment,
                                state=state, robot_state=asdict(arm), observation=payload,
                                sensor_times=sensor_times, request_id=request_id,
                                human_action=human, human_requested_action=human_requested,
                                policy_action=policy_preview['action'] if policy_preview else None,
                                policy_raw_action=candidate.get('raw_action', candidate['action']) if candidate else None,
                                policy_source=policy_source, policy_preview=policy_preview,
                                policy_observation_age_s=publish_start-candidate['observation_time'] if candidate else None,
                                pair_valid=pair_valid,
                                pair_invalid_reason=(None if pair_valid else
                                    'not_human' if arb.mode != 'human' else
                                    'no_policy_candidate' if candidate is None else policy_preview['rejection']),
                                requested_action=requested, published_action=published,
                                published_pose=dispatcher.last_robot_action_values,
                                commanded_gripper_m=dispatcher._last_gripper_width_cmd,
                                spacemouse_axes=axes.tolist(), spacemouse_buttons=buttons,
                                hil_id=keyed.hil_id if keyed_enabled else None,
                                hil_phase=keyed.phase if keyed_enabled else 'none', **stuck_fields)
                    if start_recovery:
                        if correction is not None:
                            correction.clear()
                        report = recovery.start(observation_time, state, published)
                        arb.switch('recovery' if recovery.status == 'retreating' else 'recovery_hold')
                        policy_limiter.reset()
                        # Old responses retain the previous generation and cannot be accepted.
                        tactile_report=detection.get('tactile',{})
                        triggers=(['state_action_stuck'] if detection.get('state_action',detection).get('stuck') else [])
                        if tactile_report.get('stuck'):
                            triggers.extend(tactile_report.get('trigger_reasons') or ['tactile:'+str(tactile_report.get('reason'))])
                        trigger_details=dict(reasons=triggers,
                            tracking_error_m=detection.get('tracking_error_m'),
                            position_excursion_m=detection.get('position_excursion_m'),
                            tactile_exceeded_blocks=tactile_report.get('exceeded_blocks'),
                            tactile_slip=tactile_report.get('slip'))
                        episode.add('stuck_recovery', tick=tick, time=observation_time, event='started',
                                    trigger_reason='+'.join(triggers),trigger_details=trigger_details, **report)
                        log.warning('[RECOVERY trigger] reason=%s details=%s', '+'.join(triggers), trigger_details)
                        log.warning('[RECOVERY] %s: %s target_xyz=%s moving_chunk=%s target_source=%s; p resumes after hold',
                                    recovery.status, recovery.reason, None if recovery.target is None else recovery.target[:3].tolist(),
                                    report['last_moving_chunk'], report['target_source'])
                    if resume_after_correction:
                        arb.switch('policy')
                        arb.track(published)
                        resume_request_generation = arb.generation
                        resume_time = time.monotonic()
                        resume_not_before = resume_time + ccfg['resume_delay_s']
                        episode.add('policy_resume', tick=tick, event='accepted',
                                    time=resume_time, not_before=resume_not_before,
                                    generation=arb.generation, source='correction_completed')
                        correction.clear()
                        correction_motion.reset()
                        recovery.reset()
                        stuck_detector.reset()
                        policy_limiter.reset()
                        human_controller.reset()
                        # Preserve any SpaceMouse event arriving during the last dispatch.
                        log.info('[CORRECTION] completed; hold %.3fs before fresh observation/request', ccfg['resume_delay_s'])
                    tick += 1
                    next_tick += 1./inf.fps
                    delay = next_tick-time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_tick = time.monotonic()
                # Capture ended at r/q. Home does not observe/log ticks or send predictions.
                # Do not delay homing for an in-flight inference response.
                stuck_detector.close()
                ended_at=time.monotonic()
                if reset_requested:
                    home()
                log.info('[HIL结束] 回位完成：%.2fs',time.monotonic()-ended_at)
                phase=time.monotonic()
                response = worker.drain()
                if response is not None:
                    episode.add('prediction', **response, after_capture_end=True)
                if hil.get('server_capture', False):
                    worker.end_episode()
                log.info('[HIL结束] 等待推理及server结束确认：%.2fs',time.monotonic()-phase)
                phase=time.monotonic()
                episode.seal()
                log.info('[HIL结束] 记录落盘完成：%.2fs',time.monotonic()-phase)
                from HIL_PI.outcome import choose_episode_decision
                choice = choose_episode_decision(episode, keyboard)
                if choice is None:
                    break
                phase=time.monotonic()
                path = episode.decide(choice)
                log.info('[HIL结束] 本地整理完成：%.2fs',time.monotonic()-phase)
                phase=time.monotonic()
                if hil.get('server_capture', False):
                    from HIL_PI.server_recording import sync_decision
                    result = sync_decision(path, worker)
                    log.info('[HIL结束] server提交完成：%.2fs %s',time.monotonic()-phase,result)
                log.info('数据决定完成：%s', path)
                episode = None
                if not reset_requested:
                    break
    finally:
        # Stop first, then flush/close. No home or gripper-open side effects on exit.
        if robot is not None and robot.is_connected():
            try:
                robot.stop()
            except Exception:
                log.exception('Stop failed during cleanup')
        if 'stuck_detector' in locals():
            stuck_detector.close()
        if worker is not None and getattr(worker, 'hil_session', None):
            try:
                worker.drain()
                worker.end_episode()
            except Exception:
                log.exception('Server episode remains pending; recover via saved transaction')
        for obj, method in ((episode, 'seal'), (correction, 'close'), (worker, 'close'), (mouse, 'close'),
                            (cameras, 'close'), *((r, 'stop') for r in rings),
                            (sensors, 'close_all'), (gripper, 'disconnect'), (robot, 'disconnect')):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception:
                    log.exception('Cleanup failed: %s', method)
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description='Flexiv SpaceMouse HIL; default check-only, no hardware/network access')
    parser.add_argument('config', nargs='?', default=str(ROOT/'Config/flexiv_pi_cylinder_rtc_hil.yaml'))
    parser.add_argument('--prompt', default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true', help='Open robot/cameras/HID; Enter explicitly starts homing/inference')
    mode.add_argument('--check-only', action='store_true', help='Validate configuration/import availability only (default)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cfg, raw = check_config(args.config)
    if not args.live:
        deps = {m: importlib.util.find_spec(m) is not None for m in ('hid', 'numpy', 'zmq', 'cv2', 'flexivrdk', 'pyrealsense2')}
        print('CHECK ONLY: no robot/HID/camera/server connection. Dependencies:', deps)
        print('Config valid:', args.config, '\nServer:', cfg.inference.server)
        if not all(deps.values()):
            raise SystemExit('Missing live dependency; install the required robot, camera and HID dependencies')
        return
    prompt = args.prompt if args.prompt is not None else raw['inference'].get('prompt', '')
    run_live(args.config, cfg, raw, prompt)
