# -*- coding: utf-8 -*-
"""Local HDMI console. No HTTP server or desktop window system is required."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import io
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
from engine import Engine, Cancelled
from pages import WIDTH, HEIGHT, PAGES, render
from mouse_input import MouseInput
from video_output import configure_video, encode_rgb, read_video_config
from gateway_protocol import GatewayReceiver


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value), encoding='utf-8')
    os.replace(str(tmp), str(path))


class Console:
    def __init__(self, args):
        # Register video hierarchy drivers BEFORE parsing the Overlay.
        from pynq.lib.video import VideoMode
        from pynq.lib.video.hierarchies import VideoOut
        from vib_driver import VibAccel
        self.args = args
        self.output = Path(args.evidence)
        self.output.mkdir(parents=True, exist_ok=True)
        self.acc = VibAccel(args.bitstream, args.coef)
        self.engine = Engine(self.acc, args.demo, args.model, args.coef)
        self.gateway = None
        self.gateway_enabled = False
        self.gateway_site = None
        self.gateway_frame_offset = 0
        self.gateway_hold = False
        self.gateway_seen = {}
        if args.gateway_port:
            self.gateway = GatewayReceiver(args.gateway_host, args.gateway_port)
            self.gateway.start()
            self.gateway_enabled = True
            print('GATEWAY LISTENING %s:%d' % (args.gateway_host, self.gateway.port), flush=True)
        self.dataset_choices = [name for name in ('demo_set.npz', 'dataset_test_3ch.npz') if Path(name).exists()]
        if args.demo not in self.dataset_choices:
            self.dataset_choices.insert(0,args.demo)
        self.dataset_index = self.dataset_choices.index(args.demo)
        self.pending_dataset = None
        self.display = self.acc.ol.hdmi_out
        if not isinstance(self.display, VideoOut):
            raise RuntimeError('Overlay HDMI hierarchy was not recognized as VideoOut')
        self.video_config = configure_video(self.display, self.acc.ol.axi_vdma, WIDTH, HEIGHT)
        print('HDMI CONFIG: ' + json.dumps(self.video_config), flush=True)
        self.gpio = self.acc.ol.axi_gpio_buttons
        self.mouse = MouseInput(WIDTH, HEIGHT)
        self.cursor_dirty = False
        self.base_image = None
        self.last_paint = 0
        self.capture_once = False
        self.padding_x = int(WIDTH * args.safe_margin)
        self.padding_y = int(HEIGHT * args.safe_margin)
        self.content_width = WIDTH - 2 * self.padding_x
        self.content_height = HEIGHT - 2 * self.padding_y
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.render_pool = ThreadPoolExecutor(max_workers=1)
        self.render_future = None
        self.render_token = -1
        self.events = queue.Queue()
        self.future = None
        self.cancel = threading.Event()
        self.page = args.page - 1
        self.condition = 0
        self.offset = 0
        self.filter_mode = 0
        # Static pages are the responsive default. KEY4 or the bottom mouse
        # button explicitly starts continuous waveform replay.
        self.playing = False
        self.last_tick = 0.0
        self.dirty = True
        self.generation = 0
        self.active_generation = 0
        self.requested_job = False
        self.stopping = False
        self.raw_last = int(self.gpio.channel1.read()) & 15
        self.stable_raw = self.raw_last
        self.changed_at = time.monotonic()
        resources = json.loads(Path(args.resources).read_text(encoding='utf-8'))
        self.state = {'frame_count': len(self.engine.data), 'source': str(args.demo),
                      'resources': resources, 'busy': False, 'progress': (0, 0), 'error': None,
                      'selected_dataset':self.dataset_choices[self.dataset_index],
                      'bitstream_short':hashlib.sha256(Path(args.bitstream).read_bytes()).hexdigest()[:8],
                      'source_mode': 'gateway' if self.gateway else 'file',
                      'presentation_mode': bool(args.presentation_mode),
                      'gateway_port': self.gateway.port if self.gateway else None,
                      'gateway_sites': [], 'gateway_site': None}
        self.enter_page(self.page)

    def action(self, action, value=None):
        if action == 'next':
            self.enter_page(self.page + 1)
        elif action == 'previous':
            self.enter_page(self.page - 1)
        elif action == 'page':
            self.enter_page(int(value) - 1)
        elif action in ('sample', 'key3'):
            if self.page == 7:
                if self.gateway and self.gateway_enabled:
                    sites = self.gateway.sites()
                    if sites:
                        if self.gateway_site not in sites:
                            self.gateway_site = sites[0]
                        else:
                            self.gateway_site = sites[(sites.index(self.gateway_site) + 1) % len(sites)]
                        self.gateway_frame_offset = 0
                        self.gateway_hold = False
                        self.state['gateway_site'] = self.gateway_site
                        self.state['gateway_sites'] = sites
                        self.state['notice'] = '已选择网关站点 ' + self.gateway_site
                        self.dirty = True
                        return
                self.dataset_index = (self.dataset_index + 1) % len(self.dataset_choices)
                self.state['selected_dataset'] = self.dataset_choices[self.dataset_index]
                self.dirty = True
                return
            if self.gateway and self.gateway_enabled and self.page in (0, 1, 2, 3, 4):
                snapshot = self.gateway.snapshot(self.gateway_site, self.gateway_frame_offset)
                if snapshot is not None:
                    max_offset = max(0, snapshot['count'] - 1)
                    self.gateway_frame_offset = (self.gateway_frame_offset + 1) % (max_offset + 1)
                    self.gateway_hold = True
                    self.playing = False
                    self.state['gateway_frame_offset'] = self.gateway_frame_offset
                    self.state['gateway_hold'] = True
                    self.state['notice'] = '查看站点 %s 的第 %d 个缓存帧' % (
                        snapshot['site_id'], self.gateway_frame_offset + 1)
                    self.cancel.set()
                    self.generation += 1
                    self.requested_job = True
                    self.dirty = True
                    return
            elif self.page == 1:
                self.filter_mode = (self.filter_mode + 1) % 2
            elif self.page in (5, 6):
                self.state.pop('accuracy' if self.page == 5 else 'performance', None)
                self.cancel.set()
                self.generation += 1
                self.state['busy'] = False
                self.requested_job = False
                self.dirty = True
                return
            else:
                self.condition = (self.condition + 1) % len(self.engine.classes)
                self.offset = 0
            self.cancel.set()
            self.generation += 1
            self.requested_job = True
            self.dirty = True
        elif action in ('run', 'key4'):
            if self.gateway and self.gateway_enabled and self.page in (0, 1, 2, 3, 4):
                if self.gateway_hold:
                    self.gateway_hold = False
                    self.gateway_frame_offset = 0
                    self.state.update(gateway_hold=False, gateway_frame_offset=0,
                                      notice='已恢复查看最新网关帧')
                    self.playing = self.page in (0, 1, 2, 3, 4)
                    self.requested_job = True
                    self.dirty = True
                    return
                if self.page in (0, 1, 2, 3, 4):
                    self.playing = not self.playing
                self.requested_job = True
                self.dirty = True
                return
            if self.page in (0, 1, 2, 3, 4):
                self.playing = not self.playing
                self.requested_job = self.playing
                self.dirty = True
            elif self.page == 7:
                if self.gateway:
                    if self.gateway_enabled:
                        self.gateway_enabled = False
                        self.state.update(source_mode='file', source=self.engine.source,
                                          frame_count=len(self.engine.data),
                                          notice='已切换文件评估模式')
                    elif self.state.get('selected_dataset') != self.engine.source:
                        self.pending_dataset = self.dataset_choices[self.dataset_index]
                        self.state['notice'] = '正在应用文件数据集'
                    else:
                        self.gateway_enabled = True
                        self.state.update(source_mode='gateway', notice='已切换 PC 网关模式')
                    self.dirty = True
                    return
                self.pending_dataset = self.dataset_choices[self.dataset_index]
                self.cancel.set()
                self.generation += 1
                self.dirty = True
            elif not self.future or self.future.done():
                self.requested_job = True
                self.dirty = True
        elif action == 'pause':
            self.playing = False
            self.dirty = True
        elif action == 'resume':
            self.playing = True
            self.requested_job = True
        elif action == 'quit':
            self.stopping = True
        elif action == 'export':
            result = {k:self.state[k] for k in ('source','resources','diagnosis','accuracy','performance')
                      if k in self.state}
            result.update(video=self.video_config, bitstream=self.state['bitstream_short'])
            save_json(self.output/'export_latest.json',result)
            self.state['notice'] = '已保存 evidence/export_latest.json'
            self.dirty = True
        elif action == 'dataset':
            if value not in self.dataset_choices:
                raise ValueError('Dataset is not in the installed presets')
            self.pending_dataset = value
            self.cancel.set()
            self.generation += 1
        elif action == 'capture':
            self.capture_once = True
            self.dirty = True
        elif action == 'calibrate':
            self.state['calibrate'] = True if value is None else bool(value)
            self.dirty = True
        elif action == 'color_order':
            if value in ('RGB', 'RBG', 'BGR', 'BRG', 'GBR', 'GRB'):
                self.args.color_order = value
                self.dirty = True

    def enter_page(self, page):
        self.cancel.set()
        self.generation += 1
        self.page = int(page) % len(PAGES)
        self.state['calibrate'] = False
        self.state['error'] = None
        self.state['busy'] = False
        self.state['progress'] = (0, 0)
        self.dirty = True
        self.requested_job = self.page in (0, 1, 2, 3, 4)
        if self.gateway and self.gateway_enabled and self.page == 4:
            self.playing = True
        print('PAGE %d: %s' % (self.page + 1, PAGES[self.page]), flush=True)

    def apply_dataset(self):
        if self.pending_dataset is None or self.future is not None:
            return
        selected, self.pending_dataset = self.pending_dataset, None
        try:
            self.engine = Engine(self.acc, selected, self.args.model, self.args.coef)
            self.args.demo = selected
            self.condition = self.offset = 0
            self.dataset_index = self.dataset_choices.index(selected)
            for key in ('sample','filter','diagnosis','accuracy','performance'):
                self.state.pop(key,None)
            self.state.update(source=selected, frame_count=len(self.engine.data),
                              selected_dataset=selected, notice='数据已应用',error=None)
            self.enter_page(self.page)
        except Exception as exc:
            self.state['error'] = str(exc)
        self.dirty = True

    def start_job(self):
        if self.future is not None:
            return
        self.cancel = threading.Event()
        cancel, generation, page = self.cancel, self.generation, self.page
        self.active_generation = generation
        self.requested_job = False
        self.state['busy'] = True
        self.state['error'] = None
        self.state['progress'] = (0, 0)
        condition, offset, filter_mode = self.condition, self.offset, self.filter_mode
        self.offset += 1
        gateway_snapshot = self.gateway.snapshot(self.gateway_site, self.gateway_frame_offset) if self.gateway and self.gateway_enabled else None
        if self.gateway and self.gateway_enabled and page == 4:
            if gateway_snapshot is None or len(gateway_snapshot['frames']) < 8:
                self.requested_job = False
                self.state['busy'] = False
                self.state['notice'] = '等待同一站点累计 8 帧后开始连续诊断'
                self.last_tick = time.monotonic()
                self.dirty = True
                return

        def progress(done, total):
            self.events.put((generation, done, total))

        def work():
            if gateway_snapshot is not None and page in (0, 2, 3):
                packet = gateway_snapshot['latest']
                return 'sample', self.engine.live_frame_job(packet['frame'], gateway_snapshot['site_id'],
                                                            packet['sequence'], cancel)
            if gateway_snapshot is not None and page == 4:
                if len(gateway_snapshot['frames']) < 8:
                    raise ValueError('网关诊断需要先接收同一站点的 8 帧')
                packet = gateway_snapshot['latest']
                return 'diagnosis', self.engine.live_diagnosis_job(
                    gateway_snapshot['frames'], gateway_snapshot['site_id'], packet['sequence'], cancel)
            if page in (0, 2, 3):
                return 'sample', self.engine.frame_job(condition, offset, cancel)
            if page == 1:
                return 'filter', self.engine.filter_job(filter_mode, cancel, offset)
            if page == 4:
                return 'diagnosis', self.engine.diagnosis_job(condition, offset, cancel)
            if page == 5:
                return 'accuracy', self.engine.accuracy_job(cancel, progress)
            if page == 6:
                return 'performance', self.engine.performance_job(cancel, progress)
            return None, None

        self.future = self.pool.submit(work)
        self.last_tick = time.monotonic()
        self.dirty = True

    def collect(self):
        if self.gateway and self.gateway_enabled:
            sites = self.gateway.sites()
            preferred = self.gateway.selected_site()
            if preferred in sites and preferred != self.gateway_site:
                self.gateway_site = preferred
                self.gateway_frame_offset = 0
                self.gateway_hold = False
                self.state.update(gateway_site=preferred, gateway_frame_offset=0,
                                  gateway_hold=False, notice='PC 已选择 HDMI 站点 ' + preferred)
                self.dirty = True
            if sites != self.state.get('gateway_sites'):
                self.state['gateway_sites'] = sites
                if self.gateway_site not in sites:
                    self.gateway_site = sites[0] if sites else None
                    self.state['gateway_site'] = self.gateway_site
                self.dirty = True
            packet = self.gateway.snapshot(self.gateway_site)
            if packet is not None and not self.gateway_hold:
                site = packet['site_id']
                sequence = packet['sequence']
                if self.gateway_seen.get(site) != sequence:
                    self.gateway_seen[site] = sequence
                    self.state['gateway_site'] = site
                    self.state['gateway_latest_sequence'] = sequence
                    self.state['gateway_frame_offset'] = self.gateway_frame_offset
                    self.state['gateway_hold'] = False
                    self.state['gateway_frame_count'] = packet['count']
                    self.state['source'] = 'gateway:%s' % site
                    self.state['frame_count'] = packet['count']
                    self.dirty = True
        while not self.events.empty():
            generation, done, total = self.events.get_nowait()
            if generation == self.generation:
                self.state['progress'] = (done, total)
                self.dirty = True
        if self.future is None or not self.future.done():
            return
        generation = self.active_generation
        try:
            name, result = self.future.result()
            if generation == self.generation and name:
                self.state[name] = result
                if name != 'sample':
                    save_json(self.output / (name + '.json'), result)
                if name != 'sample':
                    print('DONE %s; hardware frames=%d' % (name, self.acc.frames_completed), flush=True)
        except Cancelled:
            pass
        except Exception as exc:
            traceback.print_exc()
            if generation == self.generation:
                self.state['error'] = str(exc)
                self.playing = False
        finally:
            self.future = None
            self.state['busy'] = False
            self.dirty = True

    def buttons(self):
        now = time.monotonic()
        raw = int(self.gpio.channel1.read()) & 15
        if raw != self.raw_last:
            self.raw_last = raw
            self.changed_at = now
        elif raw != self.stable_raw and now - self.changed_at >= .055:
            edge = (~raw) & self.stable_raw & 15
            self.stable_raw = raw
            for bit, action in [(1, 'previous'), (2, 'next'), (4, 'key3'), (8, 'key4')]:
                if edge & bit:
                    print('BUTTON ' + action, flush=True)
                    self.action(action)
                    break

    def command_file(self):
        path = Path(self.args.command)
        if not path.exists():
            return
        try:
            command = json.loads(path.read_text(encoding='utf-8'))
            path.unlink()
            self.action(command['action'], command.get('value'))
        except (ValueError, KeyError, OSError) as exc:
            print('Command:', exc, flush=True)

    def mouse_events(self):
        old_names = list(self.mouse.names)
        moved, actions = self.mouse.poll()
        if self.mouse.names != old_names:
            self.state['mouse_names'] = list(self.mouse.names)
            self.dirty = True
            print('USB mouse: ' + ', '.join(self.mouse.names), flush=True)
        self.cursor_dirty |= moved
        for action, point in actions:
            if action != 'click':
                self.action(action)
                continue
            x, y = point
            x = (x - self.padding_x) * WIDTH / float(self.content_width)
            y = (y - self.padding_y) * HEIGHT / float(self.content_height)
            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                continue
            if 69 <= y <= 113:
                self.enter_page(int(x // 160))
            elif self.page == 7 and 698 <= x <= 1218 and 565 <= y <= 604:
                self.action('export')
            elif 670 <= y < HEIGHT:
                if x < 230:
                    self.action('previous')
                elif x < 448:
                    self.action('next')
                elif x < 744:
                    self.action('key3')
                elif x < 1115:
                    self.action('key4')

    def schedule_render(self):
        if not self.dirty or self.render_future is not None:
            return
        token = self.generation
        snapshot = dict(self.state)
        snapshot['playing'] = self.playing
        self.render_token = token
        self.render_future = self.render_pool.submit(render, self.page, snapshot, self.args.safe_margin)
        self.dirty = False

    def collect_render(self):
        if self.render_future is None or not self.render_future.done():
            return
        future = self.render_future
        self.render_future = None
        try:
            image = future.result()
            if self.render_token == self.generation:
                self.base_image = image
                self.present(save_status=True)
            else:
                self.dirty = True
        except Exception as exc:
            self.state['error'] = str(exc)
            self.dirty = True

    def present(self, save_status=False):
        self.state['playing'] = self.playing
        self.state['gpio_raw'] = int(self.gpio.channel1.read())
        self.state['vdma_status'] = int(self.acc.ol.axi_vdma.read(4))
        params = self.acc.ol.ip_dict['axi_gpio_buttons'].get('parameters', {})
        self.state['video_status'] = int(self.gpio.channel2.read()) if int(params.get('C_IS_DUAL', 0)) else None
        if self.base_image is None:
            return
        img = self.base_image.copy()
        if self.mouse.devices:
            x, y = self.mouse.x, self.mouse.y
            draw = ImageDraw.Draw(img)
            draw.polygon([(x, y), (x, y+23), (x+6, y+17), (x+12, y+28),
                          (x+17, y+25), (x+11, y+15), (x+21, y+15)],
                         fill='white', outline='black')
        frame = self.display.newframe()
        np.copyto(frame, encode_rgb(img, self.args.color_order))
        self.display.writeframe(frame)
        # Evidence images are the exact RGB source sent to HDMI, not screenshots of the monitor.
        if self.args.capture or self.capture_once:
            img.save(str(self.output / ('page_%02d.png' % (self.page + 1))))
            img.save(str(self.output / 'current.png'))
            self.capture_once = False
        if save_status:
            save_json(self.output / 'status.json', {
            'page': self.page + 1, 'name': PAGES[self.page], 'busy': self.state['busy'],
            'playing': self.playing, 'error': self.state.get('error'),
            'gpio_raw': self.state['gpio_raw'], 'vdma_status': self.state['vdma_status'],
            'video_status': self.state['video_status'], 'frames_completed': self.acc.frames_completed,
            'mouse_names': list(self.mouse.names), 'mouse_position': [self.mouse.x, self.mouse.y],
            'color_order': self.args.color_order, 'safe_margin': self.args.safe_margin,
            'source':self.state['source'], 'frame_count':self.state['frame_count'],
            'source_mode': self.state.get('source_mode','file'),
            'gateway_port': self.state.get('gateway_port'),
            'gateway_sites': self.state.get('gateway_sites',[]),
            'gateway_site': self.state.get('gateway_site'),
            'gateway_latest_sequence': self.state.get('gateway_latest_sequence'),
            'gateway_frame_offset': self.state.get('gateway_frame_offset', 0),
            'gateway_hold': self.gateway_hold,
            'selected_dataset':self.state['selected_dataset'],
            'video_config': self.video_config,
                'progress': self.state['progress'], 'completed_results': [k for k in ['filter', 'diagnosis', 'accuracy', 'performance'] if k in self.state]})
        self.cursor_dirty = False
        self.last_paint = time.monotonic()

    def run(self):
        print('HDMI CONSOLE READY', flush=True)
        try:
            while not self.stopping:
                self.buttons()
                self.mouse_events()
                self.command_file()
                self.collect()
                self.apply_dataset()
                self.collect_render()
                now = time.monotonic()
                continuous = self.playing and self.page in (0, 1, 2, 3, 4) and now - self.last_tick >= self.args.interval
                if self.future is None and (self.requested_job or continuous):
                    self.start_job()
                if self.dirty:
                    self.schedule_render()
                if self.cursor_dirty and now - self.last_paint >= .05:
                    self.present(save_status=False)
                time.sleep(.025)
        finally:
            self.cancel.set()
            self.pool.shutdown(wait=True)
            self.render_pool.shutdown(wait=True)
            self.display.close()
            self.mouse.close()
            if self.gateway:
                self.gateway.close()
            self.acc.close()


def main():
    # The ALINX image starts system services in the C locale (Python 3.6 ASCII).
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name)
        if getattr(stream, 'buffer', None) is not None and str(stream.encoding).lower() != 'utf-8':
            setattr(sys, name, io.TextIOWrapper(stream.buffer, encoding='utf-8',
                                               errors='replace', line_buffering=True))
    parser = argparse.ArgumentParser()
    parser.add_argument('--bitstream', default='vib_accel.bit')
    parser.add_argument('--coef', default='fir_coef_q15.npy')
    parser.add_argument('--model', default='svm_model_multich.npz')
    parser.add_argument('--demo', default='dataset_test_3ch.npz')
    parser.add_argument('--resources', default='resources.json')
    parser.add_argument('--evidence', default='evidence')
    parser.add_argument('--command', default='command.json')
    parser.add_argument('--interval', type=float, default=.5)
    parser.add_argument('--page', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--safe-margin', type=float, default=.04)
    parser.add_argument('--color-order', choices=['BGR','RGB','BRG','RBG','GRB','GBR'], default='BGR')
    parser.add_argument('--gateway-host', default='0.0.0.0')
    parser.add_argument('--gateway-port', type=int, default=0,
                        help='listen for PC multi-site frames; 0 keeps local file mode')
    parser.add_argument('--presentation-mode', action='store_true',
                        help='hide all-page navigation and control footer for filming')
    args = parser.parse_args()
    if not 0 <= args.safe_margin <= .15:
        parser.error('--safe-margin must be between 0 and 0.15')
    os.chdir(str(Path(__file__).resolve().parent))
    app = Console(args)

    def stop(signum, frame):
        app.stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    app.run()


if __name__ == '__main__':
    main()
