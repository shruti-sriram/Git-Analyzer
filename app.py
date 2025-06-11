import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import requests
import re
from urllib.parse import urlparse
import base64

import chromadb
from chromadb.config import Settings
import ollama
from nicegui import ui, app
import aiohttp

import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import ast

@dataclass
class RepoFile:
    path: str
    content: str
    size: int
    type: str

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float

@dataclass
class ComparisonResult:
    model: str
    response: str
    response_time: float
    token_count: int
    error: Optional[str] = None

selected_model = None 
input_section = None
result_section = None

class GitHubRepoProcessor:
    def __init__(self):
        self.api_base = "https://api.github.com"
        
    async def fetch_repo_info(self, repo_url: str) -> Dict[str, Any]:
        pattern = r'github\.com/([^/]+)/([^/]+)'
        match = re.search(pattern, repo_url)
        if not match:
            raise ValueError("Invalid GitHub URL format")

        owner, repo = match.groups()
        repo = repo.replace('.git', '')

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            repo_info_url = f"{self.api_base}/repos/{owner}/{repo}"
            async with session.get(repo_info_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch repository info: {response.status}")
                repo_info = await response.json()

            contents_url = f"{self.api_base}/repos/{owner}/{repo}/contents"
            repo_files = await self._fetch_repo_contents(session, contents_url)

        return {
            'info': repo_info,
            'files': repo_files
        }

    async def _fetch_repo_contents(self, session: aiohttp.ClientSession, url: str, path: str = "") -> List[RepoFile]:
        files = []
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return files
                contents = await response.json()
                tasks = []
                items_to_fetch = []

                for item in contents:
                    if item['type'] == 'file' and self._is_text_file(item['name']):
                        tasks.append(self._fetch_file_content(session, item['download_url']))
                        items_to_fetch.append(item)
                    elif item['type'] == 'dir' and not item['name'].startswith('.'):
                        if len(path.split('/')) < 3:
                            subdir_files = await self._fetch_repo_contents(session, item['url'], item['path'])
                            files.extend(subdir_files)

                results = await asyncio.gather(*tasks)
                for item, file_content in zip(items_to_fetch, results):
                    if file_content:
                        files.append(RepoFile(
                            path=item['path'],
                            content=file_content,
                            size=item['size'],
                            type=item['type']
                        ))
        except Exception as e:
            print(f"Error fetching contents from {url}: {e}")
        return files

    def _is_text_file(self, filename: str) -> bool:
        text_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
            '.html', '.css', '.scss', '.sass', '.less', '.xml', '.json',
            '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
            '.md', '.txt', '.rst', '.tex', '.sh', '.bat', '.ps1',
            '.sql', '.r', '.R', '.m', '.pl', '.lua', '.vim'}
        no_ext_files = {'README', 'LICENSE', 'CHANGELOG', 'CONTRIBUTING', 'Dockerfile', 'Makefile', 'Jenkinsfile', 'Vagrantfile'}
        file_ext = Path(filename).suffix.lower()
        file_name = Path(filename).name
        return file_ext in text_extensions or file_name in no_ext_files

    async def _fetch_file_content(self, session: aiohttp.ClientSession, download_url: str) -> Optional[str]:
        try:
            async with session.get(download_url) as response:
                if response.status == 200:
                    content_bytes = await response.read()
                    return content_bytes.decode('utf-8')
        except Exception as e:
            print(f"Error fetching file content: {e}")
        return None

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client(Settings(persist_directory="./chroma_db"))
        self.collection = None
        
    def create_collection(self, name: str):
        try:
            self.collection = self.client.get_collection(name)
        except:
            self.collection = self.client.create_collection(name)

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        if self.collection:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_text: str, n_results: int = 5) -> Dict:
        if self.collection:
            return self.collection.query(query_texts=[query_text], n_results=n_results)
        return {"documents": [], "metadatas": []}

class LLMManager:
    def __init__(self):
        self.available_models = []
        self._load_available_models()
    
    def _load_available_models(self):
        models_response = ollama.list()
        self.available_models = [model.model for model in models_response['models']]
        print(f"Parsed models: {self.available_models}")
    
    async def generate_response(self, model: str, prompt: str, context: str = "") -> ComparisonResult:
        start_time = time.time()
        try:
            if model not in self.available_models:
                end_time = time.time()
                return ComparisonResult(
                    model=model,
                    response="",
                    response_time=end_time - start_time,
                    token_count=0,
                    error=f"Model '{model}' not available. Please pull it manually if needed."
                )

            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
            response = ollama.generate(model=model, prompt=full_prompt, stream=False)
            end_time = time.time()
            response_text = response.get('response', str(response))
            token_count = len(response_text.split()) + len(full_prompt.split())
            return ComparisonResult(
                model=model,
                response=response_text,
                response_time=end_time - start_time,
                token_count=token_count
            )
        except Exception as e:
            end_time = time.time()
            return ComparisonResult(
                model=model,
                response="",
                response_time=end_time - start_time,
                token_count=0,
                error=str(e)
            )

class RepositoryAnalyzer:
    def __init__(self):
        self.repo_processor = GitHubRepoProcessor()
        self.vector_store = VectorStore()
        self.llm_manager = LLMManager()
        self.current_repo = None
        self.chat_history = []
        
    async def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        try:
            repo_data = await self.repo_processor.fetch_repo_info(repo_url)
            self.current_repo = repo_data
            repo_name = repo_data['info']['name']
            self.vector_store.create_collection(f"repo_{repo_name}")
            documents, metadatas, ids = [], [], []

            for i, file in enumerate(repo_data['files']):
                documents.append(f"File: {file.path}\n\n{file.content}")
                metadatas.append({'path': file.path, 'size': file.size, 'type': file.type})
                ids.append(f"file_{i}")

            if documents:
                self.vector_store.add_documents(documents, metadatas, ids)

            summary = await self._generate_summary(repo_data)
            flow = await self._generate_flow(repo_data)

            return {
                'summary': summary,
                'flow': flow,
                'repo_info': repo_data['info'],
                'file_count': len(repo_data['files'])
            }
        except Exception as e:
            return {'error': str(e)}

    async def _generate_summary(self, repo_data: Dict) -> str:
        repo_info = repo_data['info']
        files = repo_data['files']
        context = f"""
Repository: {repo_info['name']}
Description: {repo_info.get('description', 'No description')}
Language: {repo_info.get('language', 'Multiple')}
Stars: {repo_info.get('stargazers_count', 0)}
Files analyzed: {len(files)}

Key files structure:
"""
        file_lines = [f"- {file.path} ({file.size} bytes)" for file in files[:10]]
        context += "\n" + "\n".join(file_lines)
        prompt = "Based on the repository information provided, create a comprehensive summary of what this repository does, its main purpose, and key features."
        if self.llm_manager.available_models:
            result = await self.llm_manager.generate_response(selected_model, prompt, context)
            return result.response
        return "Summary generation not available - no LLM models found."

    async def _generate_flow(self, repo_data: Dict) -> str:
        files = repo_data['files']
        context = "Repository file structure and flow:\n"
        file_structure = {}
        for file in files:
            dir_name = str(Path(file.path).parent)
            file_structure.setdefault(dir_name, []).append(file.path)
        dir_lines = []
        for directory, files_list in file_structure.items():
            file_names = "\n".join([f"  - {Path(f).name}" for f in files_list[:5]])
            dir_lines.append(f"\nDirectory: {directory}\n{file_names}")
        context += "\n".join(dir_lines)
        prompt = "Based on the file structure provided, describe the architecture and flow of this repository. Explain how the components interact and the overall structure."
        if self.llm_manager.available_models:
            result = await self.llm_manager.generate_response(selected_model, prompt, context)
            return result.response
        return "Flow analysis not available - no LLM models found."

    async def chat_with_repo(self, question: str, model: str) -> str:
        """Chat about the repository"""
        if not self.current_repo:
            return "No repository loaded. Please analyze a repository first."

        query_result = self.vector_store.query(question, n_results=3)
        context = "\n\n".join(query_result['documents'][0]) if query_result['documents'] else ""

        result = await self.llm_manager.generate_response(model, question, context)

        self.chat_history.append(ChatMessage("user", question, time.time()))
        self.chat_history.append(ChatMessage("assistant", result.response, time.time()))

        return result.response
    
    async def compare_models(self, question: str, models: List[str]) -> List[ComparisonResult]:
        """Compare responses from multiple models"""
        if not self.current_repo:
            return []

        query_result = self.vector_store.query(question, n_results=3)
        context = "\n\n".join(query_result['documents'][0]) if query_result['documents'] else ""

        tasks = [self.llm_manager.generate_response(model, question, context) for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        comparison_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                comparison_results.append(ComparisonResult(
                    model=models[i],
                    response="",
                    response_time=0,
                    token_count=0,
                    error=str(result)
                ))
            else:
                comparison_results.append(result)

        return comparison_results

# ---------------- FRONTEND SECTION ---------------- #

analyzer = RepositoryAnalyzer()
current_panel = "input"
repo_analysis_result = None
chat_model = None

def create_sidebar():
    with ui.column().classes('w-100 bg-gray-100 h-screen p-4 justify-center items-center'):
        ui.label('Git Repository Analyzer').classes('text-xl font-bold mb-4')
        try:
            ollama.list()
            ui.label(f"Ollama Connected").classes('text-sm text-green-600 mb-2')
        except Exception as e:
            ui.label("Ollama Not Connected").classes('text-sm text-red-600 mb-2')
            ui.label("Please start Ollama service").classes('text-xs text-gray-500 mb-2')

        with ui.column().classes('space-y-2'):
            model_list = analyzer.llm_manager.available_models
            global selected_model
            if model_list and selected_model is None:
                selected_model = model_list[0]

            ui.select(
                options=model_list,
                value=selected_model,
                label='LLM Model',
                on_change=lambda e: set_model(e.value)
            ).classes('w-full mb-4')

            ui.button('Input', on_click=lambda: switch_panel('input')).classes('w-full')
            ui.button('Chat', on_click=lambda: switch_panel('chat')).classes('w-full')
            ui.button('Compare', on_click=lambda: switch_panel('compare')).classes('w-full')
            ui.button('Graph', on_click=lambda: switch_panel('graph')).classes('w-full')


def set_model(value):
    global selected_model
    selected_model = value

def switch_panel(panel_name: str):
    global current_panel
    current_panel = panel_name
    main_content.clear()

    if panel_name == 'input':
        create_input_panel()
    elif panel_name == 'chat':
        create_chat_panel()
    elif panel_name == 'compare':
        create_compare_panel()
    elif panel_name == 'graph':
        create_graph_panel()

def create_input_panel():
    global input_section, result_section
    main_content.clear()

    with main_content:
        input_section = ui.column().classes('w-full items-center justify-center')
        with input_section:
            ui.label('Repository Analysis').classes('text-2xl font-bold mb-4')
            url_input = ui.input('GitHub Repository URL', placeholder='https://github.com/username/repository').classes('w-2/3 mb-4')
            # ui.button('Analyze Repository', on_click=lambda: analyze_repo(url_input.value)).classes('mb-4')
            ui.button('Analyze Repository', on_click=lambda: asyncio.create_task(analyze_repo(url_input.value)))


        result_section = ui.column().classes('w-full items-center')

async def analyze_repo(repo_url: str):
    global repo_analysis_result

    if not repo_url:
        ui.notify('Please enter a repository URL', type='warning')
        return

    input_section.set_visibility(False)  

    result_section.clear()
    with result_section:
        ui.label('Analyzing repository...').classes('text-lg')
        ui.spinner(size='lg')

    try:
        result = await analyzer.analyze_repository(repo_url)
        repo_analysis_result = result

        result_section.clear()
        if 'error' in result:
            ui.label(f'Error: {result["error"]}').classes('text-red-500')
        else:
            with result_section:
                ui.label('Analysis Complete!').classes('text-2xl font-bold text-green-600 mb-4')
                ui.label('Repository Information').classes('text-xl font-bold mb-2')
                ui.label(f"Name: {result['repo_info']['name']}").classes('mb-1')
                ui.label(f"Description: {result['repo_info'].get('description', 'No description')}").classes('mb-1')
                ui.label(f"Language: {result['repo_info'].get('language', 'Multiple')}").classes('mb-1')
                ui.label(f"Stars: {result['repo_info'].get('stargazers_count', 0)}").classes('mb-1')
                ui.label(f"Files analyzed: {result['file_count']}").classes('mb-4')
                ui.label('Summary').classes('text-xl font-bold mb-2')
                ui.label(result['summary']).classes('mb-4 p-4 bg-gray-50 rounded')
                ui.label('Repository Flow').classes('text-xl font-bold mb-2')
                ui.label(result['flow']).classes('mb-4 p-4 bg-gray-50 rounded')
    except Exception as e:
        result_section.clear()
        ui.label(f'Error analyzing repository: {str(e)}').classes('text-red-500')

def create_chat_panel():
    with main_content:
        ui.label('Chat with Repository').classes('text-2xl font-bold mb-4')
        if not repo_analysis_result:
            ui.label('Please analyze a repository first in the Input panel.').classes('text-orange-500 mb-4')
            return
        
        ui.label(f"Model: {selected_model}").classes("mb-4")

        chat_history_area = ui.column().classes('h-48 w-2/3 overflow-y-auto border p-4 mb-4 bg-gray-50')
        chat_input = ui.input('Ask a question about the repository...').classes('w-2/3 flex-grow')

        with ui.column().classes('w-2/3 items-center'):
            chat_input.classes('flex-grow mr-2')
            ui.button('Send', on_click=lambda: send_chat_message(
                chat_input.value, selected_model, chat_history_area, chat_input
            ))


        display_chat_history(chat_history_area)

def display_chat_history(container):
    container.clear()
    for message in analyzer.chat_history[-10:]:
        with container:
            role_label = "You" if message.role == "user" else "Assistant"
            ui.label(f"{role_label}: {message.content}").classes(
                'mb-2 p-2 rounded ' + 
                ('bg-blue-100' if message.role == 'user' else 'bg-green-100')
            )

async def send_chat_message(question: str, model: str, history_area, input_field):
    if not question or not model:
        ui.notify('Please enter a question and select a model', type='warning')
        return

    input_field.value = ''
    with history_area:
        ui.label(f"You: {question}").classes('mb-2 p-2 rounded bg-blue-100')
        ui.label("Assistant: Thinking...").classes('mb-2 p-2 rounded bg-gray-100')

    try:
        response = await analyzer.chat_with_repo(question, model)
        display_chat_history(history_area)
    except Exception as e:
        ui.label(f"Error: {str(e)}").classes('text-red-500')

def create_compare_panel():
    with main_content:
        ui.label('Compare Models').classes('text-2xl font-bold mb-4')
        if not repo_analysis_result:
            ui.label('Please analyze a repository first in the Input panel.').classes('text-orange-500 mb-4')
            return

        ui.label('Select models to compare:').classes('mb-2')
        model_checkboxes = []
        with ui.column().classes('mb-4'):
            for model in analyzer.llm_manager.available_models:
                checkbox = ui.checkbox(model, value=True)
                model_checkboxes.append((model, checkbox))

        question_input = ui.input('Question to compare across models').classes('w-full mb-4')
        ui.button('Compare Models', on_click=lambda: compare_models(question_input.value, model_checkboxes)).classes('mb-4')

        global compare_results_area
        compare_results_area = ui.column().classes('w-full')

def create_graph_panel():
    with main_content:
        ui.label('File Dependency Graph').classes('text-2xl font-bold mb-4')
        if not repo_analysis_result:
            ui.label('Please analyze a repository first in the Input panel.').classes('text-orange-500 mb-4')
            return

        file_list = analyzer.current_repo.get('files', [])
        graph = nx.DiGraph()
        module_map = {Path(f.path).stem: f for f in file_list}

        for file in file_list:
            fname = Path(file.path).stem
            if not file.content:
                continue

            try:
                # 🔹 Python Imports
                if file.path.endswith('.py'):
                    tree = ast.parse(file.content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imp = alias.name.split('.')[0]
                                if imp in module_map:
                                    graph.add_edge(fname, imp)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imp = node.module.split('.')[0]
                                if imp in module_map:
                                    graph.add_edge(fname, imp)

                # 🔹 JavaScript / TypeScript Imports
                elif file.path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                    imports = re.findall(r'import .* from [\'"]([^\'"]+)[\'"]', file.content)
                    for imp in imports:
                        imp_stem = Path(imp).stem
                        if imp_stem in module_map:
                            graph.add_edge(fname, imp_stem)

                # 🔹 Java Imports
                elif file.path.endswith('.java'):
                    imports = re.findall(r'import ([a-zA-Z0-9_.]+);', file.content)
                    for imp in imports:
                        imp_stem = imp.split('.')[-1]
                        if imp_stem in module_map:
                            graph.add_edge(fname, imp_stem)

                # 🔹 C/C++ Includes
                elif file.path.endswith(('.c', '.cpp', '.h')):
                    includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', file.content)
                    for inc in includes:
                        inc_stem = Path(inc).stem
                        if inc_stem in module_map:
                            graph.add_edge(fname, inc_stem)

                # Add more language-specific logic here as needed

            except Exception as e:
                print(f"Failed to parse {file.path}: {e}")

        # 🔍 Plot the graph
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='lightblue', arrows=True, ax=ax)
        ax.set_title("Cross-Language File Dependency Graph", fontsize=14)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        ui.image(f'data:image/png;base64,{encoded}').classes('w-full')

def generate_model_comparison_chart(results):
    models = [res.model for res in results if not res.error]
    response_times = [res.response_time for res in results if not res.error]
    token_counts = [res.token_count for res in results if not res.error]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax2 = ax1.twinx()
    width = 0.4
    x = range(len(models))

    ax1.bar([i - width/2 for i in x], response_times, width=width, label='Response Time (s)')
    ax2.bar([i + width/2 for i in x], token_counts, width=width, label='Token Count', color='orange')

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Response Time (s)')
    ax2.set_ylabel('Token Count')

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_title('Model Comparison: Response Time and Token Count')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded

async def compare_models(question: str, model_checkboxes):
    if not question:
        ui.notify('Please enter a question', type='warning')
        return

    selected_models = [model for model, checkbox in model_checkboxes if checkbox.value]
    if not selected_models:
        ui.notify('Please select at least one model', type='warning')
        return

    compare_results_area.clear()
    with compare_results_area:
        ui.label('Comparing models...').classes('text-lg')
        ui.spinner(size='lg')

    try:
        results = await analyzer.compare_models(question, selected_models)
        compare_results_area.clear()
        with compare_results_area:
            ui.label('Comparison Results').classes('text-2xl font-bold mb-4')
            for result in results:
                with ui.card().classes('w-full mb-4'):
                    ui.label(f'Model: {result.model}').classes('text-xl font-bold')
                    if result.error:
                        ui.label(f'Error: {result.error}').classes('text-red-500')
                    else:
                        ui.label(f'Response Time: {result.response_time:.2f}s').classes('text-sm text-gray-600')
                        ui.label(f'Estimated Tokens: {result.token_count}').classes('text-sm text-gray-600')
                        ui.label('Response:').classes('font-bold mt-2')
                        ui.label(result.response).classes('p-2 bg-gray-50 rounded')

            # 🟢 Add chart after showing textual results
            chart_base64 = generate_model_comparison_chart(results)
            ui.label("Visual Summary").classes('text-xl font-bold mt-6 mb-2')
            ui.image(f'data:image/png;base64,{chart_base64}').classes('w-full')

    except Exception as e:
        compare_results_area.clear()
        ui.label(f'Error comparing models: {str(e)}').classes('text-red-500')

with ui.row().classes('w-full h-screen'):
    create_sidebar()
    main_content = ui.column().classes('flex-1 p-6 overflow-y-auto justify-center self-center items-center')
    create_input_panel()

ui.add_head_html('''
<style>
    .nicegui-content {
        padding: 0 !important;
    }
</style>
''')

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title='RAG Repository Analyzer', favicon='🔍', dark=False)
