import ast
import copy
import hashlib
import json
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

NOTEBOOK = Path(__file__).with_name('Untitled0.ipynb')


def cells():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']


def extract_function(cell_index, name, namespace):
    tree = ast.parse(''.join(cells()[cell_index]['source']))
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), f'<cell {cell_index}:{name}>', 'exec'), namespace)
    return namespace[name]


class NotebookControlsTests(unittest.TestCase):
    def test_dependency_setup_preserves_colab_constraints_and_checks_loaded_versions(self):
        source = ''.join(cells()[5]['source'])
        compile('\n'.join(source.splitlines()[1:]), '<dependency setup cell>', 'exec')
        self.assertIn('"requests==2.32.4"', source)
        self.assertIn('"huggingface-hub>=1.23.0,<2.0.0"', source)
        self.assertNotIn(' -U ', source)
        self.assertIn('metadata.requires(distribution)', source)
        self.assertIn('sys.modules.get(module_name)', source)
        self.assertLess(source.index('from huggingface_hub import HfApi, hf_hub_download'), source.index('依存ライブラリの準備完了'))

    def test_notebook_schema_and_code_syntax(self):
        nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.assertEqual(len(nb['cells']), 37)
        self.assertEqual(len({cell.get('id') for cell in nb['cells'] if cell.get('id')}), sum(bool(cell.get('id')) for cell in nb['cells']))
        helper_bytes = Path(__file__).with_name('continue_colab_bridge.py').read_bytes().replace(b'\r\n', b'\n')
        helper_hash = hashlib.sha256(helper_bytes).hexdigest()
        rerun_cell = nb['cells'][36]
        rerun_source = ''.join(rerun_cell['source'])
        self.assertEqual(rerun_cell['id'], 'rerun-refresh')
        self.assertIn('# @title 通常再実行', rerun_source.splitlines()[0])
        self.assertEqual(hashlib.sha256(rerun_source.encode('utf-8')).hexdigest(), '2c231f956f06af4d5b921c39241fb3900cbffe37ce1ad62b5f98d32bfde7a722')
        bridge_cell = ''.join(nb['cells'][34]['source'])
        self.assertIn(helper_hash, bridge_cell)
        self.assertIn("CONTINUE_BRIDGE_SHA256 = '" + helper_hash + "'", bridge_cell)
        snapshot_bytes = Path(__file__).with_name('colab_refresh_snapshot.json').read_bytes()
        self.assertIn(hashlib.sha256(snapshot_bytes).hexdigest(), rerun_source)
        snapshot = json.loads(snapshot_bytes)
        self.assertEqual(snapshot['model_alias'], 'ternary-bonsai2-27b-abliterated-pq2-gguf')
        for index in (13, 15, 19, 20, 21, 34):
            self.assertEqual(snapshot['cells'][str(index)], ''.join(nb['cells'][index]['source']))
        self.assertIn("REPO_ID = 'Hikari07jp/Ternary-Bonsai-2-27B-Abliterated-GGUF'", ''.join(nb['cells'][6]['source']))
        self.assertIn("MODEL_REVISION = '187cabcc22baf475832376dd033c6283d309d5f1'", ''.join(nb['cells'][6]['source']))
        self.assertIn("LLAMA_COMMIT = 'adfffbe41b2cabcd51fff326ab045662265062bb'", ''.join(nb['cells'][7]['source']))
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                self.assertNotIn('execution_count', cell)
                self.assertNotIn('outputs', cell)
        for i in (11, 13, 14, 15, 17, 19, 20, 21, 34, 36):
            source = ''.join(nb['cells'][i].get('source', []))
            compile(source, f'<notebook cell {i}>', 'exec')

    def test_both_backend_processes_and_endpoints_are_retained_when_switching(self):
        text_process = SimpleNamespace(pid=101, poll=lambda: None)
        image_process = SimpleNamespace(pid=202, poll=lambda: None, args=['python', '/mock/ComfyUI/main.py'])
        response = SimpleNamespace(status_code=200)
        ns = {
            'GPU_INFERENCE_LOCK': threading.Lock(),
            'GPU_INFERENCE_BLOCKED': False,
            'Path': Path,
            'IMAGE_SERVER_PROCESS': image_process,
            'IMAGE_BASE_URL': 'http://127.0.0.1:8188',
            'server_process': text_process,
            'requests': SimpleNamespace(get=Mock(return_value=response), post=Mock(return_value=response)),
            'validate_image_object_info': Mock(),
            'IMAGE_PORT': 8188,
            'IMAGE_SERVER_LOG_HANDLE': None,
            'IMAGE_COMFY_DIR': Path('/mock/ComfyUI'),
            'IMAGE_HOST': '127.0.0.1',
            'BASE_URL': 'http://127.0.0.1:9000',
            'MODEL_ALIAS': 'ternary-bonsai2-27b-abliterated-pq2-gguf',
            'chat_history': [],
            'PRIVATE_ROUTE_BASE_URL': 'http://127.0.0.1:7000',
            'PRIVATE_ROUTE_TOKEN': 'test-token',
            'create_chat_ui': Mock(return_value=('input', 'send', 'reset', 'output')),
            'widgets': SimpleNamespace(VBox=lambda x: x),
            'display': Mock(),
            '_bonsai_health': Mock(return_value=True),
        }
        matches = extract_function(20, '_matches_comfy_process', ns)
        self.assertTrue(matches(image_process))
        start_image = extract_function(20, 'start_image_backend', ns)
        self.assertEqual(start_image(), 'http://127.0.0.1:8188')
        restore_chat = extract_function(21, 'restore_bonsai', ns)
        restore_chat()
        self.assertIs(ns['server_process'], text_process)
        self.assertIs(ns['IMAGE_SERVER_PROCESS'], image_process)
        ns['validate_image_object_info'].assert_called_once()

    def test_shared_gpu_lock_rejects_overlap_and_recovers_after_error(self):
        source = ''.join(cells()[13]['source'])
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_gpu_serialized']
        self.assertEqual(len(nodes), 1)
        ns = {'GPU_INFERENCE_LOCK': threading.Lock(), 'GPU_INFERENCE_BLOCKED': False}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(nodes[0])], type_ignores=[])), '<lock helper>', 'exec'), ns)
        run_serialized = ns['run_gpu_serialized']
        entered = threading.Event()
        release = threading.Event()
        worker = threading.Thread(target=lambda: run_serialized(lambda: (entered.set(), release.wait(2))))
        worker.start()
        self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(RuntimeError, 'GPUは別の推論'):
            run_serialized(lambda: None)
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        with self.assertRaises(ValueError):
            run_serialized(lambda: (_ for _ in ()).throw(ValueError('expected')))
        self.assertEqual(run_serialized(lambda: 'recovered'), 'recovered')



    def test_prompt_budget_reserves_output_and_safety_for_context(self):
        namespace = {'PROMPT_SAFETY_TOKENS': 64}
        budget = extract_function(13, 'prompt_token_budget', namespace)
        self.assertEqual(budget(4096, 512), 3520)
        self.assertLessEqual(budget(4096, 512) + 512 + 64, 4096)
        self.assertEqual(budget(4096, 1024), 3008)
        with self.assertRaises(ValueError):
            budget(4096, 4096)
        for index, function_name in ((13, 'generate_reply'), (15, '_generate_unlocked')):
            tree = ast.parse(''.join(cells()[index]['source']))
            function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function_name)
            self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'prompt_token_budget' for n in ast.walk(function)))

    def test_stop_button_respects_active_lock_then_stops_owned_backends(self):
        lock = threading.Lock()
        clear_image = Mock()
        stop_text = Mock()
        requests = SimpleNamespace(post=Mock())
        ns = {
            'GPU_INFERENCE_LOCK': lock,
            'GPU_INFERENCE_BLOCKED': False,
            '_clear_image_process': clear_image,
            'stop_server': stop_text,
            'IMAGE_SERVER_PROCESS': object(),
            'server_process': object(),
            'requests': requests,
            'PRIVATE_ROUTE_BASE_URL': 'http://127.0.0.1:7000',
            'PRIVATE_ROUTE_TOKEN': 'test-token',
            'print': Mock(),
        }
        clear_image.side_effect = lambda: ns.update(IMAGE_SERVER_PROCESS=None)
        stop_text.side_effect = lambda: ns.update(server_process=None)
        stop = extract_function(21, 'stop_all_backends', ns)
        lock.acquire()
        stop(None)
        self.assertFalse(clear_image.called)
        self.assertFalse(stop_text.called)
        lock.release()
        stop(None)
        clear_image.assert_called_once()
        stop_text.assert_called_once()
        self.assertFalse(lock.locked())

    def test_stopping_only_comfy_keeps_uncertain_inference_blocked(self):
        ns = {
            'GPU_INFERENCE_LOCK': threading.Lock(),
            'GPU_INFERENCE_BLOCKED': True,
            '_clear_image_process': Mock(),
            'requests': SimpleNamespace(post=Mock()),
            'PRIVATE_ROUTE_BASE_URL': 'http://127.0.0.1:7000',
            'PRIVATE_ROUTE_TOKEN': 'test-token',
            'print': Mock(),
        }
        stop_image = extract_function(21, 'stop_image_backend', ns)
        stop_image()
        self.assertTrue(ns['GPU_INFERENCE_BLOCKED'])

    def test_ui_routes_both_inference_paths_through_the_shared_lock(self):
        chat_tree = ast.parse(''.join(cells()[15]['source']))
        create = next(n for n in chat_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_chat_ui')
        chat_generate = next(n for n in ast.walk(create) if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run_gpu_serialized' for n in ast.walk(chat_generate)))
        image_tree = ast.parse(''.join(cells()[21]['source']))
        create_image = next(n for n in image_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_image_ui')
        image_run = next(n for n in ast.walk(create_image) if isinstance(n, ast.FunctionDef) and n.name == 'run')
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run_gpu_serialized' for n in ast.walk(image_run)))


    def _run_mocked_rerun_cell(self, *, busy=False, unhealthy=False, missing=False):
        current = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        source = ''.join(current['cells'][36]['source'])
        snapshot_bytes = Path(__file__).with_name('colab_refresh_snapshot.json').read_bytes()
        snapshot = json.loads(snapshot_bytes)
        helper_bytes = Path(__file__).with_name('continue_colab_bridge.py').read_bytes()
        lock = threading.Lock()
        if busy:
            lock.acquire()
        history = [{'role': 'user', 'content': 'keep this'}]
        text_proc = SimpleNamespace(pid=101, args=[
            '/content/qwen38_work/llama-prism-b10743/build/bin/llama-server', '-m',
            '/content/qwen38_gguf/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf', '--alias', 'ternary-bonsai2-27b-abliterated-pq2-gguf'
        ], poll=Mock(return_value=None), terminate=Mock(), wait=Mock())
        image_proc = SimpleNamespace(pid=202, args=[
            'python', '/content/qwen_image_work/ComfyUI/main.py'
        ], poll=Mock(return_value=None), terminate=Mock(), wait=Mock())
        calls = []
        self._last_rerun_calls = calls

        class Response:
            def __init__(self, url):
                self.url = url
                self.status_code = 503 if unhealthy else 200
                self.content = snapshot_bytes if url.endswith('/colab_refresh_snapshot.json') else helper_bytes

            def json(self):
                return {'data': [{'id': 'ternary-bonsai2-27b-abliterated-pq2-gguf'}]}

            def raise_for_status(self):
                if self.status_code != 200:
                    raise RuntimeError('mock HTTP error')

        def get(url, **_kwargs):
            self.assertTrue(lock.locked(), 'refresh must hold the shared inference lock during network checks')
            calls.append(url)
            return Response(url)

        class Widget:
            def __init__(self, *args, **kwargs):
                self.value = kwargs.get('value', None)
                self.disabled = False
                self.children = args

            def on_click(self, callback):
                self.callback = callback

            def observe(self, callback, **_kwargs):
                self.observer = callback

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        requests_module = ModuleType('requests')
        requests_module.get = get
        requests_module.post = Mock(side_effect=AssertionError('refresh must not POST'))
        psutil_module = ModuleType('psutil')
        psutil_module.Process = lambda pid: SimpleNamespace(cmdline=lambda: text_proc.args if pid == 101 else image_proc.args)
        widgets_module = ModuleType('ipywidgets')
        for name in ('Button', 'Checkbox', 'FileUpload', 'Layout', 'Output', 'Textarea'):
            setattr(widgets_module, name, Widget)
        widgets_module.HBox = widgets_module.VBox = lambda children: list(children)
        ipython_module = ModuleType('IPython')
        ipython_module.__path__ = []
        ipython_display = ModuleType('IPython.display')
        ipython_display.display = Mock()
        ipython_display.clear_output = Mock()
        subprocess_module = ModuleType('subprocess')
        subprocess_module.Popen = Mock(side_effect=AssertionError('refresh must not start processes'))
        subprocess_module.run = Mock(side_effect=AssertionError('refresh must not build/install'))
        subprocess_module.TimeoutExpired = subprocess.TimeoutExpired

        namespace = {
            '__name__': '__rerun_test__',
            'GPU_INFERENCE_LOCK': lock,
            'GPU_INFERENCE_BLOCKED': False,
            'LLAMA_CTX_SIZE': 4096,
            'chat_history': history,
            'BASE_URL': 'http://text.local',
            'MODEL_ALIAS': 'ternary-bonsai2-27b-abliterated-pq2-gguf',
            'SERVER_BIN': Path(text_proc.args[0]),
            'model_path': Path(text_proc.args[2]),
            'server_process': text_proc,
            'IMAGE_COMFY_DIR': Path('/content/qwen_image_work/ComfyUI'),
            'IMAGE_BASE_URL': 'http://image.local',
            'IMAGE_SERVER_PROCESS': None if missing else image_proc,
            'CONTINUE_BRIDGE': None,
            'widgets': widgets_module,
        }
        original_modules = {name: sys.modules.get(name) for name in (
            'requests', 'psutil', 'ipywidgets', 'IPython', 'IPython.display', 'subprocess'
        )}
        executed_cells = []
        optional_cell = []
        import builtins

        def dispatch_cell(code, target_globals):
            if code.co_filename.endswith('cell-34>'):
                optional_cell.append(code)
                return
            executed_cells.append(code.co_filename)
            builtins.exec(code, target_globals, target_globals)

        namespace['exec'] = dispatch_cell
        sys.modules.update({
            'requests': requests_module,
            'psutil': psutil_module,
            'ipywidgets': widgets_module,
            'IPython': ipython_module,
            'IPython.display': ipython_display,
            'subprocess': subprocess_module,
        })
        try:
            exec(compile(source, '<rerun cell>', 'exec'), namespace)
            self.assertIs(namespace['GPU_INFERENCE_LOCK'], lock)
            self.assertIs(namespace['server_process'], text_proc)
            self.assertIs(namespace['IMAGE_SERVER_PROCESS'], image_proc)
            self.assertIs(namespace['chat_history'], history)
            self.assertEqual(history, [{'role': 'user', 'content': 'keep this'}])
            self.assertIsNone(namespace['CONTINUE_BRIDGE'])
            self.assertFalse(lock.locked())
            subprocess_module.Popen.assert_not_called()
            subprocess_module.run.assert_not_called()
            requests_module.post.assert_not_called()
            text_proc.terminate.assert_not_called()
            image_proc.terminate.assert_not_called()
            self.assertFalse(any('/chat/completions' in url for url in calls))
            self.assertFalse(any('huggingface' in url.lower() or '/releases/' in url for url in calls))
            self.assertEqual(executed_cells, [f'<verified-notebook-cell-{i}>' for i in (13, 15, 19, 20, 21)])
            self.assertEqual(len(optional_cell), 1)
            return calls, namespace, snapshot['cells']['34']
        finally:
            if lock.locked():
                lock.release()
            for name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


    def test_rerun_refresh_executes_real_lightweight_cells_and_preserves_runtime_state(self):
        calls, ns, bridge_source = self._run_mocked_rerun_cell()
        self.assertTrue(any(url.endswith('/colab_refresh_snapshot.json') for url in calls))
        self.assertTrue(callable(ns['run_gpu_serialized']))
        self.assertTrue(callable(ns['create_chat_ui']))
        self.assertTrue(callable(ns['create_image_ui']))
        self.assertTrue(callable(ns['stop_all_backends']))
        bridge_tree = ast.parse(bridge_source)
        top_level_calls = [n for n in bridge_tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
        self.assertFalse(any(isinstance(n.value.func, ast.Attribute) and n.value.func.attr in ('start_tunnel', 'stop', 'stop_accepting') for n in top_level_calls))

    def test_rerun_refresh_refuses_busy_unhealthy_or_incomplete_runtime_before_notebook_fetch(self):
        for options in ({'busy': True}, {'unhealthy': True}, {'missing': True}):
            with self.subTest(options=options), self.assertRaises(RuntimeError):
                self._run_mocked_rerun_cell(**options)
            self.assertFalse(any(url.endswith('/colab_refresh_snapshot.json') for url in self._last_rerun_calls))


if __name__ == '__main__':
    unittest.main()

