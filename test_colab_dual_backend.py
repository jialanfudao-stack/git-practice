import ast
import copy
import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
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
    def test_notebook_schema_and_code_syntax(self):
        nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.assertEqual(len(nb['cells']), 33)
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                self.assertNotIn('execution_count', cell)
                self.assertNotIn('outputs', cell)
        for i in (11, 13, 14, 15, 17, 19, 20, 21):
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
            'MODEL_ALIAS': 'text-model',
            'chat_history': [],
            'PRIVATE_ROUTE_BASE_URL': 'http://127.0.0.1:7000',
            'PRIVATE_ROUTE_TOKEN': 'test-token',
            'create_chat_ui': Mock(return_value=('input', 'send', 'reset', 'output')),
            'widgets': SimpleNamespace(VBox=lambda x: x),
            'display': Mock(),
            '_huihui_health': Mock(return_value=True),
        }
        matches = extract_function(20, '_matches_comfy_process', ns)
        self.assertTrue(matches(image_process))
        start_image = extract_function(20, 'start_image_backend', ns)
        self.assertEqual(start_image(), 'http://127.0.0.1:8188')
        restore_chat = extract_function(21, 'restore_huihui', ns)
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
            'requests': requests,
            'PRIVATE_ROUTE_BASE_URL': 'http://127.0.0.1:7000',
            'PRIVATE_ROUTE_TOKEN': 'test-token',
            'print': Mock(),
        }
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

    def test_ui_routes_both_inference_paths_through_the_shared_lock(self):
        chat_tree = ast.parse(''.join(cells()[15]['source']))
        create = next(n for n in chat_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_chat_ui')
        chat_generate = next(n for n in ast.walk(create) if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run_gpu_serialized' for n in ast.walk(chat_generate)))
        image_tree = ast.parse(''.join(cells()[21]['source']))
        create_image = next(n for n in image_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_image_ui')
        image_run = next(n for n in ast.walk(create_image) if isinstance(n, ast.FunctionDef) and n.name == 'run')
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run_gpu_serialized' for n in ast.walk(image_run)))


if __name__ == '__main__':
    unittest.main()

