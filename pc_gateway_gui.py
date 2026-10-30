"""Windows PC control panel for the AX7020 multi-site vibration gateway.

The PC window is a configuration/transport tool. The diagnostic result remains
on the board HDMI display. It uses only tkinter and NumPy.
"""
from __future__ import print_function
import argparse
import queue
import socket
import threading
import time
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'outputs' / 'AX7020_HDMI'))
from gateway_protocol import DEFAULT_FS_HZ, pack_frame, pack_site_select


def default_dataset():
    """Work both in the development tree and the flat submission project."""
    for base in (HERE, HERE / 'outputs' / 'AX7020_HDMI'):
        candidate = base / 'dataset_test_3ch.npz'
        if candidate.is_file():
            return candidate
    return HERE / 'dataset_test_3ch.npz'

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    tk = None


def load_frames(path):
    archive = np.load(str(path), allow_pickle=False)
    try:
        key = 'X' if 'X' in archive.files else 'X_test_i16'
        data = np.asarray(archive[key])
        if data.ndim != 3 or data.shape[1:] != (3, 1024) or data.dtype != np.int16:
            raise ValueError('数据必须是 int16、(N,3,1024)')
        return data
    finally:
        archive.close()


class Sender(object):
    def __init__(self, board, port, dataset, sites, focus_site, interval, callback):
        self.board, self.port = board, int(port)
        self.dataset = dataset
        self.sites = sites
        self.focus_site = focus_site
        self.interval = float(interval)
        self.callback = callback
        self.stop_event = threading.Event()
        self.thread = None
        self.sent = 0
        self.sequence = 0
        self.lock = threading.Lock()

    def start(self):
        if self.thread and self.thread.is_alive():
            return False
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.stop_event.set()

    def set_focus_site(self, site):
        with self.lock:
            self.focus_site = str(site)
        self.callback(('focus', str(site)))

    def run(self):
        sock = None
        try:
            data = load_frames(self.dataset)
            sock = socket.create_connection((self.board, self.port), timeout=8.)
            sock.settimeout(8.)
            with self.lock:
                sock.sendall(pack_site_select(self.focus_site))
            self.callback(('connected', len(data)))
            index = 0
            while not self.stop_event.is_set():
                with self.lock:
                    focus_site = self.focus_site
                if focus_site:
                    # Re-send the selection at the start of each cycle so the
                    # board follows the PC choice even after reconnects.
                    sock.sendall(pack_site_select(focus_site))
                for offset, site in enumerate(self.sites):
                    frame = data[(index + offset * max(1, len(data) // len(self.sites))) % len(data)]
                    sock.sendall(pack_frame(site, self.sequence, frame))
                    self.sequence += 1
                    self.sent += 1
                    self.callback(('sent', self.sent, self.sequence, site, frame))
                    if self.stop_event.is_set():
                        break
                index += 1
                time.sleep(max(0., self.interval))
        except Exception as exc:
            self.callback(('error', str(exc)))
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            self.callback(('stopped', self.sent))


class GatewayGui(object):
    BG = '#0c141c'
    PANEL = '#172832'
    PANEL2 = '#203541'
    FG = '#e7f2f2'
    DIM = '#9db4bd'
    GREEN = '#42d7b5'
    BLUE = '#6db8ef'
    GOLD = '#f0bc60'
    RED = '#f58b86'

    def __init__(self, root):
        self.root = root
        self.root.title('AMD FPGA 振动数据网关')
        self.root.geometry('1120x720')
        self.root.minsize(980, 640)
        self.root.configure(bg=self.BG)
        self.events = queue.Queue()
        self.sender = None
        self.last_frame = None
        self.sent = 0
        self.started_at = None
        self._style()
        self._build()
        self.root.after(80, self._poll)

    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        style.configure('TButton', font=('Microsoft YaHei UI', 11), padding=(13, 7))
        style.configure('Accent.TButton', foreground='#061713', background=self.GREEN,
                        font=('Microsoft YaHei UI', 11, 'bold'), padding=(13, 7))
        style.configure('TEntry', padding=5)
        style.configure('TLabel', background=self.PANEL, foreground=self.FG,
                        font=('Microsoft YaHei UI', 10))
        style.configure('Title.TLabel', background=self.BG, foreground=self.FG,
                        font=('Microsoft YaHei UI', 22, 'bold'))
        style.configure('Sub.TLabel', background=self.BG, foreground=self.DIM,
                        font=('Microsoft YaHei UI', 10))
        style.configure('Panel.TFrame', background=self.PANEL)

    def _build(self):
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill='both', expand=True, padx=22, pady=18)
        ttk.Label(outer, text='AMD FPGA 多站点振动数据网关', style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer, text='PC 负责配置与传输 · AX7020 HDMI 负责本地分析结果显示',
                  style='Sub.TLabel').pack(anchor='w', pady=(2, 15))

        main = tk.Frame(outer, bg=self.BG)
        main.pack(fill='both', expand=True)
        left = tk.Frame(main, bg=self.PANEL, width=395)
        left.pack(side='left', fill='y', padx=(0, 12))
        left.pack_propagate(False)
        right = tk.Frame(main, bg=self.PANEL, width=650)
        right.pack(side='left', fill='both', expand=True)

        self._section(left, '连接与数据源')
        form = tk.Frame(left, bg=self.PANEL)
        form.pack(fill='x', padx=18)
        self.board_var = tk.StringVar(value='192.168.2.99')
        self.port_var = tk.StringVar(value='9000')
        self.dataset_var = tk.StringVar(value=str(default_dataset()))
        self.sites_var = tk.StringVar(value='motor-A,motor-B,motor-C')
        self.interval_var = tk.StringVar(value='0.05')
        self.focus_var = tk.StringVar(value='motor-A')
        self._field(form, '板卡 IP', self.board_var)
        self._field(form, 'TCP 端口', self.port_var)
        row = tk.Frame(form, bg=self.PANEL); row.pack(fill='x', pady=5)
        ttk.Label(row, text='数据集', width=8).pack(side='left')
        ttk.Entry(row, textvariable=self.dataset_var).pack(side='left', fill='x', expand=True)
        ttk.Button(row, text='选择', command=self.choose_dataset).pack(side='left', padx=(5, 0))
        self._field(form, '站点列表', self.sites_var)
        focus_row = tk.Frame(form, bg=self.PANEL); focus_row.pack(fill='x', pady=5)
        ttk.Label(focus_row, text='HDMI站点', width=8).pack(side='left')
        self.focus_combo = ttk.Combobox(focus_row, textvariable=self.focus_var,
                                        values=('motor-A', 'motor-B', 'motor-C'), state='readonly', width=15)
        self.focus_combo.pack(side='left', fill='x', expand=True)
        self.focus_button = ttk.Button(focus_row, text='切换显示', command=self.change_focus)
        self.focus_button.pack(side='left', padx=(5, 0))
        self._field(form, '发送间隔(s)', self.interval_var)
        buttons = tk.Frame(left, bg=self.PANEL); buttons.pack(fill='x', padx=18, pady=12)
        ttk.Button(buttons, text='测试连接', command=self.test_connection).pack(side='left', expand=True, fill='x')
        self.start_button = ttk.Button(buttons, text='开始发送', style='Accent.TButton', command=self.start)
        self.start_button.pack(side='left', expand=True, fill='x', padx=(8, 0))
        self.stop_button = ttk.Button(buttons, text='停止', command=self.stop, state='disabled')
        self.stop_button.pack(side='left', expand=True, fill='x', padx=(8, 0))

        self._section(left, '发送状态')
        status = tk.Frame(left, bg=self.PANEL); status.pack(fill='x', padx=18)
        self.status_var = tk.StringVar(value='未连接')
        self.count_var = tk.StringVar(value='0 个数据包')
        self.site_var = tk.StringVar(value='最后发送站点：等待')
        self.rate_var = tk.StringVar(value='0.0 pkt/s')
        for var, color in ((self.status_var, self.GREEN), (self.count_var, self.BLUE),
                           (self.site_var, self.GOLD), (self.rate_var, self.DIM)):
            tk.Label(status, textvariable=var, bg=self.PANEL, fg=color,
                     font=('Microsoft YaHei UI', 13, 'bold')).pack(anchor='w', pady=3)
        tk.Label(left, text='提示：PC 窗口用于网关配置和调试；板卡 HDMI 显示站点请到 08 系统配置按 PL KEY3 选择。',
                 bg=self.PANEL, fg=self.DIM, justify='left', wraplength=350,
                 font=('Microsoft YaHei UI', 10)).pack(anchor='w', padx=18, pady=(18, 0))

        self._section(right, '发送预览（PC 调试画面）')
        preview = tk.Frame(right, bg=self.PANEL); preview.pack(fill='both', expand=True, padx=18, pady=(0, 10))
        self.canvases = []
        for name, color in (('DE 驱动端', self.BLUE), ('FE 风扇端', self.GREEN), ('BA 基座', self.GOLD)):
            holder = tk.Frame(preview, bg=self.PANEL2); holder.pack(fill='both', expand=True, pady=5)
            tk.Label(holder, text=name, bg=self.PANEL2, fg=color,
                     font=('Microsoft YaHei UI', 10, 'bold')).pack(anchor='w', padx=8, pady=(4, 0))
            canvas = tk.Canvas(holder, height=105, bg=self.PANEL2, highlightthickness=0)
            canvas.pack(fill='both', expand=True, padx=8, pady=(0, 5))
            self.canvases.append((canvas, color))
        self.log = tk.Text(right, height=5, bg='#0a1117', fg=self.DIM, insertbackground=self.FG,
                           relief='flat', font=('Consolas', 9), state='disabled')
        self.log.pack(fill='x', padx=18, pady=(0, 14))
        self._log('等待操作。先点击“测试连接”，再点击“开始发送”。')

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=self.PANEL, fg=self.FG,
                 font=('Microsoft YaHei UI', 13, 'bold')).pack(anchor='w', padx=18, pady=(18, 9))

    def _field(self, parent, label, variable):
        row = tk.Frame(parent, bg=self.PANEL); row.pack(fill='x', pady=5)
        ttk.Label(row, text=label, width=8).pack(side='left')
        ttk.Entry(row, textvariable=variable).pack(side='left', fill='x', expand=True)

    def _log(self, message):
        self.log.configure(state='normal')
        self.log.insert('end', time.strftime('%H:%M:%S ') + message + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def choose_dataset(self):
        path = filedialog.askopenfilename(title='选择 int16 三通道 NPZ',
                                          filetypes=[('NPZ', '*.npz'), ('All files', '*.*')])
        if path:
            self.dataset_var.set(path)

    def test_connection(self):
        try:
            with socket.create_connection((self.board_var.get().strip(), int(self.port_var.get())), timeout=3):
                self.status_var.set('连接成功')
                self._log('TCP %s:%s 连接成功' % (self.board_var.get().strip(), self.port_var.get()))
        except Exception as exc:
            self.status_var.set('连接失败')
            self._log('连接失败：' + str(exc))
            messagebox.showerror('连接失败', str(exc))

    def start(self):
        sites = [x.strip() for x in self.sites_var.get().split(',') if x.strip()]
        if not sites:
            messagebox.showerror('参数错误', '至少填写一个站点名')
            return
        try:
            load_frames(Path(self.dataset_var.get()))
            interval = float(self.interval_var.get())
            if interval <= 0:
                raise ValueError('发送间隔必须大于 0')
        except Exception as exc:
            messagebox.showerror('数据或参数错误', str(exc))
            return
        if self.focus_var.get().strip() not in sites:
            self.focus_var.set(sites[0])
        self.focus_combo.configure(values=tuple(sites))
        self.sender = Sender(self.board_var.get().strip(), int(self.port_var.get()),
                             Path(self.dataset_var.get()), sites, self.focus_var.get().strip(),
                             interval, self.events.put)
        self.sender.start()
        self.started_at = time.monotonic()
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.status_var.set('正在发送')
        self._log('开始发送站点：' + ', '.join(sites))

    def change_focus(self):
        site = self.focus_var.get().strip()
        if not site:
            return
        if self.sender and self.sender.thread and self.sender.thread.is_alive():
            self.sender.set_focus_site(site)
            self._log('已请求板卡 HDMI 显示站点：' + site)
        else:
            self._log('已设置 HDMI 站点：' + site + '；开始发送后生效')

    def stop(self):
        if self.sender:
            self.sender.stop()
        self.status_var.set('正在停止')
        self.stop_button.configure(state='disabled')

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == 'connected':
                    self.status_var.set('已连接并发送')
                    self._log('板卡已连接，数据集帧数=%d' % event[1])
                elif kind == 'sent':
                    _, count, sequence, site, frame = event
                    self.sent = count
                    self.last_frame = frame
                    self.count_var.set('%d 个数据包' % count)
                    self.site_var.set('最后发送站点：' + site)
                    elapsed = max(time.monotonic() - self.started_at, .001)
                    self.rate_var.set('%.1f pkt/s' % (count / elapsed))
                    if count == 1 or count % 75 == 0:
                        self._log('发送 %d 包，序号=%d，站点=%s' % (count, sequence, site))
                elif kind == 'focus':
                    self.site_var.set('HDMI 显示站点：' + event[1])
                elif kind == 'error':
                    self.status_var.set('发送失败')
                    self._log('发送失败：' + event[1])
                    messagebox.showerror('网关发送失败', event[1])
                elif kind == 'stopped':
                    self.status_var.set('已停止')
                    self.stop_button.configure(state='disabled')
                    self.start_button.configure(state='normal')
                    self._log('发送停止，共发送 %d 包' % event[1])
        except queue.Empty:
            pass
        if self.last_frame is not None:
            self._draw_preview(self.last_frame)
        self.root.after(80, self._poll)

    def _draw_preview(self, frame):
        for channel, (canvas, color) in enumerate(self.canvases):
            canvas.delete('all')
            width = max(canvas.winfo_width(), 200)
            height = max(canvas.winfo_height(), 80)
            values = frame[channel].astype(float)
            indices = np.linspace(0, len(values) - 1, min(width - 12, 900)).astype(int)
            values = values[indices]
            peak = max(float(np.max(np.abs(values))), 1.)
            points = []
            for i, value in enumerate(values):
                x = 6 + i * (width - 12) / max(1, len(values) - 1)
                y = height / 2. - value / peak * (height * .40)
                points.extend((x, y))
            canvas.create_line(*points, fill=color, width=1.4)
            canvas.create_line(6, height / 2., width - 6, height / 2., fill='#3b4c55')


def self_test():
    data = load_frames(default_dataset())
    print('GUI self-test PASS: dataset=%s dtype=%s frame=%s' % (data.shape, data.dtype, data[0].shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if tk is None:
        raise SystemExit('当前 Python 没有 tkinter，请使用 python.org Windows Python 或启用 Tcl/Tk')
    root = tk.Tk()
    GatewayGui(root)
    root.mainloop()


if __name__ == '__main__':
    main()
