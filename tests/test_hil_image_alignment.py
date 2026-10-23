"""Native image transport and tactile metadata; no sensor/device access."""
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from Inference.client import InferenceClient, encode_native_png
from HIL.observations import CameraWorker


@pytest.mark.parametrize('shape', [(480,640,3), (360,640,3)])
def test_shared_encoder_lossless_geometry_channels_and_range(shape):
    bgr = np.random.default_rng(7).integers(0,256,shape,dtype=np.uint8)
    wire = InferenceClient('unused')._encode_observation({'observation/image': bgr},[0.]*32)
    decoded = cv2.imdecode(np.frombuffer(wire['observation/image'],np.uint8),cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded,bgr)
    rgb = cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(rgb,bgr[:,:,::-1])
    assert decoded.min() == 0 and decoded.max() == 255


@pytest.mark.parametrize('image', [np.zeros((3,4,3),np.float32),np.zeros((3,4),np.uint8),np.zeros((3,4,4),np.uint8)])
def test_wrong_dtype_or_layout_rejected(image):
    with pytest.raises(ValueError,match='HWC BGR uint8'):
        encode_native_png(image)


def test_hil_worker_native_rgb_and_per_sensor_baseline_history():
    worker = CameraWorker.__new__(CameraWorker)
    image = np.random.default_rng(8).integers(0,256,(480,640,3),dtype=np.uint8)
    wrist = image[:360].copy()
    worker.names = ['third','wrist']
    worker.mapping = {'third':'observation/image','wrist':'observation/wrist_image'}
    worker.lock = threading.Lock()
    worker.running = True
    worker.error = None
    def read(names):
        worker.running = False
        return {'third':image,'wrist':wrist}
    worker.sensors = SimpleNamespace(read_images=read)
    worker._run()
    assert worker.error is None
    rings=[]
    for side in ['left','right']:
        rings.append(SimpleNamespace(name=side, wire_key=f'observation/{side}_tactile',
                     snapshot_timed=lambda side=side: ([side.encode()+b'1',side.encode()+b'2'],side.encode()+b'base',time.monotonic())))
    for _ in range(2):
        wire,_=worker.snapshot([0.]*32,rings,'test',history_frames=1)
        for key,expected in [('observation/image',image),('observation/wrist_image',wrist)]:
            decoded=cv2.imdecode(np.frombuffer(wire[key],np.uint8),cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(decoded,expected)
        for side in ['left','right']:
            key=f'observation/{side}_tactile'
            assert wire[key]==side.encode()+b'2'
            assert wire['extra']['tactile_history'][key]==[side.encode()+b'2']
            assert wire['extra']['tactile_baseline'][key]==side.encode()+b'base'


@pytest.mark.parametrize('name',['flexiv_vtla_hil.yaml','flexiv_vtla_hil_online.yaml','flexiv_vtla_expo_hil.yaml'])
def test_hil_camera_modes(name):
    cfg=yaml.safe_load((Path('Config')/name).read_text())
    assert len(cfg['tactile'])==2
    assert all((s['width'],s['height'])==(640,360) for s in cfg['tactile'].values())
    assert all((s['streams']['color']['width'],s['streams']['color']['height'])==(640,480)
               for s in cfg['cameras'].values())
