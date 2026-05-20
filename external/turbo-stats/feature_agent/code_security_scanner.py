import re
import ast
from typing import List, Tuple, Dict, Any

class CodeSecurityScanner:
    """Сканер безопасности Python кода"""
    
    def __init__(self):
        self.patterns = self._init_patterns()
        self.dangerous_modules = self._init_dangerous_modules()
        self.whitelisted_modules = ['math', 'datetime', 'json', 'random', 'string', 'collections', 'itertools', 'functools']
        
    def _init_patterns(self) -> List[Tuple[str, str]]:
        """Инициализация паттернов для regex проверки"""
        return [
            # Системные вызовы
            (r'\bexec\s*\(', 'Dangerous exec call'),
            (r'\beval\s*\(', 'Dangerous eval call'),
            (r'\bcompile\s*\(', 'Code compilation'),
            (r'\b__import__\s*\(', 'Dynamic import'),
            (r'\bglobals\s*\(', 'Access to global namespace'),
            (r'\blocals\s*\(', 'Access to local namespace'),
            (r'\bgetattr\s*\(', 'Dynamic attribute access'),
            (r'\bsetattr\s*\(', 'Dynamic attribute setting'),
            (r'\bdelattr\s*\(', 'Dynamic attribute deletion'),
            
            # Файловые операции
            (r'\bopen\s*\(', 'File operations not allowed'),
            (r'\b__\w+__', 'Double underscore pattern (magic methods)'),
            
            # Опасные строковые операции
            (r'\.format\s*\([^)]*\{[^}]*\{', 'Nested format strings'),
            (r'f\s*\".*\{.*\{.*\}.*\}', 'Complex nested f-strings'),
            
            # Системные команды
            (r'os\.(?:system|popen|popen2|popen3|popen4|spawn)\s*\(', 'OS system commands'),
            (r'subprocess\.(?:call|run|Popen|check_output)\s*\(', 'Subprocess execution'),
            
            # Сетевые операции
            (r'socket\.(?:socket|create_connection)\s*\(', 'Socket operations'),
            (r'urllib\.', 'URL library operations'),
            (r'http\.', 'HTTP module operations'),
            
            # Сериализация
            (r'pickle\.(?:loads|load)\s*\(', 'Pickle deserialization'),
            (r'marshal\.(?:loads|load)\s*\(', 'Marshal deserialization'),
            (r'shelve\.open\s*\(', 'Shelve database'),
            
            # Отражение и интроспекция
            (r'inspect\.', 'Code introspection'),
            (r'traceback\.', 'Traceback manipulation'),
            # (r'frame\.', 'Frame operations'),
            (r'sys\._getframe\s*\(', 'Get frame from call stack'),
            
            # Процессы и потоки
            (r'threading\.', 'Thread operations'),
            (r'multiprocessing\.', 'Multi-processing'),
            (r'ctypes\.', 'C types operations'),
            
            # Дополнительные опасные вызовы
            (r'eval\(.*\(', 'Nested eval calls'),
            (r'exec\(.*\(', 'Nested exec calls'),
            (r'input\s*\(', 'User input (potential injection)'),
            (r'breakpoint\s*\(', 'Breakpoint call'),
            (r'quit\s*\(', 'Quit call'),
            (r'exit\s*\(', 'Exit call'),
        ]
    
    def _init_dangerous_modules(self) -> List[str]:
        """Список опасных модулей"""
        return [
            'os', 'sys', 'subprocess', 'pickle', 'socket', 'requests',
            'urllib', 'ctypes', 'mmap', 'signal', 'inspect', 'traceback',
            'resource', 'gc', 'shutil', 'tempfile', 'pty', 'posix', 'nt',
            'pdb', 'bdb', 'code', 'codeop', 'zipimport', 'zipfile',
            'tarfile', 'ftplib', 'telnetlib', 'smtplib', 'poplib',
            'imaplib', 'nntplib', 'cgi', 'cgitb', 'webbrowser', 'ssl',
            'crypt', 'hashlib', 'hmac', 'secrets', 'py_compile',
            'compileall', 'imp', 'importlib', 'runpy', 'marshal',
            'shelve', 'dbm', 'sqlite3', 'xml', 'lxml', 'html',
            'multiprocessing', 'threading', 'concurrent', 'asyncio',
            'select', 'selectors', 'asyncore', 'asynchat',
        ]
    
    def scan_with_regex(self, code: str) -> List[Dict[str, str]]:
        """Сканирование кода с помощью regex паттернов"""
        issues = []
        
        for pattern, description in self.patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                issues.append({
                    'type': 'regex',
                    'description': description,
                    'pattern': pattern,
                    'match': match.group(0),
                    'line': code[:match.start()].count('\n') + 1,
                    'position': match.start()
                })
        
        return issues
    
    def scan_with_ast(self, code: str) -> List[Dict[str, Any]]:
        """Сканирование с использованием AST"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [{
                'type': 'syntax',
                'description': f'Syntax error: {str(e)}',
                'line': e.lineno if hasattr(e, 'lineno') else 0,
                'position': e.offset if hasattr(e, 'offset') else 0
            }]
        
        for node in ast.walk(tree):
            # Проверка импортов
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.dangerous_modules:
                        issues.append({
                            'type': 'import',
                            'description': f'Dangerous module import: {alias.name}',
                            'module': alias.name,
                            'line': node.lineno if hasattr(node, 'lineno') else 0,
                            'asname': alias.asname
                        })
            
            # Проверка from imports
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in self.dangerous_modules:
                    issues.append({
                        'type': 'import_from',
                        'description': f'Dangerous import from: {node.module}',
                        'module': node.module,
                        'line': node.lineno if hasattr(node, 'lineno') else 0,
                        'names': [alias.name for alias in node.names]
                    })
            
            # Проверка вызовов функций
            elif isinstance(node, ast.Call):
                # Проверка eval, exec, compile
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile', '__import__', 'globals', 'locals']:
                        issues.append({
                            'type': 'dangerous_call',
                            'description': f'Dangerous function call: {node.func.id}',
                            'function': node.func.id,
                            'line': node.lineno if hasattr(node, 'lineno') else 0
                        })
        
        return issues
    
    def check_imports_safety(self, code: str) -> List[Dict[str, Any]]:
        """Проверка безопасности импортов"""
        issues = []
        
        # Парсим все импорты
        try:
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'name': alias.name,
                            'asname': alias.asname,
                            'line': node.lineno,
                            'type': 'import'
                        })
                elif isinstance(node, ast.ImportFrom):
                    imports.append({
                        'name': node.module,
                        'asname': None,
                        'line': node.lineno,
                        'type': 'from_import',
                        'imported': [alias.name for alias in node.names]
                    })
            
            # Проверяем каждый импорт
            for imp in imports:
                module_name = imp['name']
                
                # Проверка на опасные модули
                if module_name and any(module_name.startswith(dm) for dm in self.dangerous_modules):
                    issues.append({
                        'type': 'dangerous_import',
                        'description': f'Dangerous module imported: {module_name}',
                        'module': module_name,
                        'line': imp['line'],
                        'details': imp
                    })
                
                # Проверка импортов из опасных модулей
                if imp['type'] == 'from_import' and imp['imported']:
                    if module_name and any(module_name.startswith(dm) for dm in self.dangerous_modules):
                        for imported_name in imp['imported']:
                            issues.append({
                                'type': 'dangerous_from_import',
                                'description': f'Dangerous import from {module_name}: {imported_name}',
                                'module': module_name,
                                'imported': imported_name,
                                'line': imp['line']
                            })
        
        except SyntaxError:
            pass
        
        return issues
    
    def scan_comprehensive(self, code: str) -> Dict[str, Any]:
        """Комплексное сканирование кода"""
        results = {
            'safe': True,
            'issues': [],
            'summary': {
                'total_issues': 0,
                'by_type': {},
                'by_severity': {}
            }
        }
        
        # Regex сканирование
        regex_issues = self.scan_with_regex(code)
        results['issues'].extend(regex_issues)
        
        # AST сканирование
        ast_issues = self.scan_with_ast(code)
        results['issues'].extend(ast_issues)
        
        # Проверка импортов
        import_issues = self.check_imports_safety(code)
        results['issues'].extend(import_issues)
        
        # Удаление дубликатов
        unique_issues = []
        seen = set()
        for issue in results['issues']:
            issue_key = (issue.get('type', ''), 
                        issue.get('description', ''), 
                        issue.get('line', 0))
            if issue_key not in seen:
                seen.add(issue_key)
                unique_issues.append(issue)
        
        results['issues'] = unique_issues
        results['summary']['total_issues'] = len(unique_issues)
        
        # Подсчет по типам
        for issue in unique_issues:
            issue_type = issue.get('type', 'unknown')
            results['summary']['by_type'][issue_type] = results['summary']['by_type'].get(issue_type, 0) + 1
        
        # Определение безопасности
        if unique_issues:
            results['safe'] = False
        
        return results
    
    def format_report(self, scan_results: Dict[str, Any]) -> str:
        """Форматирование отчета"""
        report = []
        
        if scan_results['issues']:
            report.append("=" * 60)
            report.append("CODE SECURITY SCAN REPORT")
            report.append("=" * 60)
            report.append(f"Overall safety: {'SAFE' if scan_results['safe'] else 'UNSAFE'}")
            report.append(f"Total issues found: {scan_results['summary']['total_issues']}")
            report.append("-" * 60)
            for i, issue in enumerate(scan_results['issues'], 1):
                report.append(f"\n{i}. [{issue.get('type', 'unknown').upper()}] {issue.get('description', 'No description')}")
            report.append("\n" + "=" * 60)
            return "\n".join(report)
        else:
            return ""
        
    
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Анализ сложности кода"""
        try:
            tree = ast.parse(code)
            
            # Подсчет различных элементов
            node_counts = {}
            for node in ast.walk(tree):
                node_type = type(node).__name__
                node_counts[node_type] = node_counts.get(node_type, 0) + 1
            
            return {
                'total_nodes': sum(node_counts.values()),
                'node_types': node_counts,
                'import_count': node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0),
                'function_count': node_counts.get('FunctionDef', 0) + node_counts.get('AsyncFunctionDef', 0),
                'class_count': node_counts.get('ClassDef', 0),
                'call_count': node_counts.get('Call', 0),
            }
        except:
            return {'error': 'Could not analyze complexity'}