import numpy as np
import pytest

from Core import ActionSpace
from Core.config_schema import CameraConfig, TactileConfig
import Inference.async_worker as async_worker_mod
from Inference.action_smoothing import TemporalActionSmoother
from Inference.async_worker import AsyncInferenceWorker, InferenceObservationSnapshot
from Inference.obs_mapping import build_camera_key_map, map_image_keys


def test_camera_mapped_key_remaps_only_inference_payload_keys():
    images = {
        "third_view": np.zeros((2, 2, 3), dtype=np.uint8),
        "left_wrist": np.ones((2, 2, 3), dtype=np.uint8),
    }
    cameras = {
        "third_view": CameraConfig(mapped_key="observation/image"),
        "left_wrist": CameraConfig(),
    }

    key_map = build_camera_key_map(cameras, enabled_cameras=["third_view", "left_wrist"])
    mapped = map_image_keys(images, key_map)

    assert sorted(mapped) == ["left_wrist", "observation/image"]
    assert mapped["observation/image"] is images["third_view"]
    assert mapped["left_wrist"] is images["left_wrist"]
    assert "third_view" in images


def test_camera_mapped_key_rejects_duplicate_payload_keys():
    cameras = {
        "cam_a": CameraConfig(mapped_key="observation/image"),
        "cam_b": CameraConfig(mapped_key="observation/image"),
    }

    with pytest.raises(ValueError, match="image sensor mapped_key conflict"):
        build_camera_key_map(cameras)


def test_tactile_mapped_key_remaps_inference_payload_keys():
    images = {
        "third_view": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist_tactile": np.ones((2, 2, 3), dtype=np.uint8),
    }
    cameras = {
        "third_view": CameraConfig(mapped_key="observation/image"),
    }
    tactile = {
        "wrist_tactile": TactileConfig(mapped_key="observation/tactile_image"),
    }

    key_map = build_camera_key_map(
        cameras,
        enabled_cameras=["third_view", "wrist_tactile"],
        tactile=tactile,
    )
    mapped = map_image_keys(images, key_map)

    assert sorted(mapped) == ["observation/image", "observation/tactile_image"]
    assert mapped["observation/image"] is images["third_view"]
    assert mapped["observation/tactile_image"] is images["wrist_tactile"]


def test_camera_and_tactile_mapped_key_conflict_is_rejected():
    cameras = {
        "third_view": CameraConfig(mapped_key="observation/image"),
    }
    tactile = {
        "wrist_tactile": TactileConfig(mapped_key="observation/image"),
    }

    with pytest.raises(ValueError, match="image sensor mapped_key conflict"):
        build_camera_key_map(cameras, tactile=tactile)


class _FakeClient:
    def __init__(self, actions=None):
        self.calls = []
        self.actions = actions or [[0.0], [1.0], [2.0]]

    def predict_chunk(self, images, state, *, prompt="", extra=None):
        self.calls.append((images, state, prompt, extra))
        return self.actions


def test_async_inference_worker_integrates_mapped_observation_chunk_once():
    client = _FakeClient()
    smoother = TemporalActionSmoother(action_space=ActionSpace.JOINT_POSITION)
    images = {"cam": np.zeros((2, 2, 3), dtype=np.uint8)}

    worker = AsyncInferenceWorker(
        client=client,
        observation_fn=lambda: InferenceObservationSnapshot(
            images=images,
            state=[0.1],
            prompt="pick",
        ),
        smoother=smoother,
        action_space=ActionSpace.JOINT_POSITION,
        overlap_steps=2,
        max_latency_steps=1,
        obs_fps=30,
    )

    assert worker.run_once() is True

    call_images, call_state, call_prompt, call_extra = client.calls[-1]
    assert call_images is images
    assert call_state == [0.1]
    assert call_prompt == "pick"
    assert call_extra is None
    assert smoother.pop_next() == pytest.approx([0.0])
    assert smoother.pop_next() == pytest.approx([1.0])
    assert smoother.pop_next() == pytest.approx([2.0])


def test_async_worker_trims_canonical_32d_actions_before_smoothing():
    raw_action = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.04] + [99.0] * 22
    client = _FakeClient(actions=[raw_action])
    smoother = TemporalActionSmoother(action_space=ActionSpace.CARTESIAN)

    worker = AsyncInferenceWorker(
        client=client,
        observation_fn=lambda: InferenceObservationSnapshot(
            images={},
            state=[0.0] * 32,
        ),
        smoother=smoother,
        action_space=ActionSpace.CARTESIAN,
        policy_format="canonical",
        canonical_dim=32,
        effective_action_dim=10,
        overlap_steps=2,
        max_latency_steps=1,
        obs_fps=30,
    )

    assert worker.run_once() is True
    assert smoother.pop_next() == pytest.approx(
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.04]
    )


def test_async_worker_drops_latency_steps_from_inference_time(monkeypatch):
    client = _FakeClient(actions=[[0.0], [1.0], [2.0], [3.0]])
    smoother = TemporalActionSmoother(action_space=ActionSpace.JOINT_POSITION)
    times = iter([10.0, 10.035])
    monkeypatch.setattr(async_worker_mod.time, "perf_counter", lambda: next(times))

    worker = AsyncInferenceWorker(
        client=client,
        observation_fn=lambda: InferenceObservationSnapshot(images={}, state=[0.0]),
        smoother=smoother,
        action_space=ActionSpace.JOINT_POSITION,
        overlap_steps=2,
        max_latency_steps=2,
        obs_fps=5,
        action_fps=100,
    )

    assert worker.run_once() is True
    assert smoother.pop_next() == pytest.approx([2.0])
    assert smoother.pop_next() == pytest.approx([3.0])
    assert smoother.pop_next() is None
