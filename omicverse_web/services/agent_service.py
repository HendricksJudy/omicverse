"""
Agent Service - OmicVerse Agent Management
==========================================
Manages OmicVerse AI agent for code generation and chat.
"""

import json
import asyncio
import threading
import logging


# Global agent instance cache
agent_instance = None
agent_config_signature = None


def get_agent_instance(config):
    """Get or create OmicVerse agent instance with caching.

    Args:
        config: Agent configuration dictionary with model, apiKey, apiBase

    Returns:
        OmicVerse Agent instance
    """
    import omicverse as ov
    global agent_instance, agent_config_signature

    if config is None:
        config = {}

    signature_payload = {
        'model': config.get('model') or 'gpt-5',
        'api_key': config.get('apiKey') or '',
        'endpoint': config.get('apiBase') or None,
    }
    signature = json.dumps(signature_payload, sort_keys=True)

    # Only recreate agent if configuration changed
    if agent_instance is None or signature != agent_config_signature:
        agent_instance = ov.Agent(
            model=signature_payload['model'],
            api_key=signature_payload['api_key'] or None,
            endpoint=signature_payload['endpoint'] or None,
            use_notebook_execution=False
        )
        agent_config_signature = signature

    return agent_instance


def run_agent_stream(agent, prompt, adata):
    """Run agent with streaming support for code generation.

    Args:
        agent: OmicVerse Agent instance
        prompt: User prompt for code generation
        adata: AnnData object for analysis

    Returns:
        Dictionary with code, llm_text, result_adata, result_shape
    """
    async def _runner():
        code = None
        result_adata = None
        result_shape = None
        llm_text = ''
        async for event in agent.stream_async(prompt, adata):
            if event.get('type') == 'llm_chunk':
                llm_text += event.get('content', '')
            elif event.get('type') == 'code':
                code = event.get('content')
            elif event.get('type') == 'result':
                result_adata = event.get('content')
                result_shape = event.get('shape')
            elif event.get('type') == 'error':
                raise RuntimeError(event.get('content', 'Agent error'))
        return {
            'code': code,
            'llm_text': llm_text,
            'result_adata': result_adata,
            'result_shape': result_shape
        }

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_container = {}
        error_container = {}

        def _run_in_thread():
            try:
                result_container['value'] = asyncio.run(_runner())
            except BaseException as exc:
                error_container['error'] = exc

        thread = threading.Thread(target=_run_in_thread, name='OmicVerseAgentRunner')
        thread.start()
        thread.join()
        if 'error' in error_container:
            raise error_container['error']
        return result_container.get('value')

    return asyncio.run(_runner())


def run_agent_chat(agent, prompt):
    """Run agent in chat mode for natural language responses.

    Args:
        agent: OmicVerse Agent instance
        prompt: User prompt for chat

    Returns:
        String response from agent
    """
    async def _runner():
        if not agent._llm:
            raise RuntimeError("LLM backend is not initialized")
        chat_prompt = (
            "You are an OmicVerse assistant. Answer in natural language only, "
            "avoid code unless explicitly requested.\n\nUser: " + prompt
        )
        return await agent._llm.run(chat_prompt)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_container = {}
        error_container = {}

        def _run_in_thread():
            try:
                result_container['value'] = asyncio.run(_runner())
            except BaseException as exc:
                error_container['error'] = exc

        thread = threading.Thread(target=_run_in_thread, name='OmicVerseAgentChatRunner')
        thread.start()
        thread.join()
        if 'error' in error_container:
            raise error_container['error']
        return result_container.get('value')

    return asyncio.run(_runner())


def agent_requires_adata(prompt):
    """Check if prompt requires adata based on keywords.

    Args:
        prompt: User prompt to analyze

    Returns:
        Boolean indicating if adata is likely required
    """
    if not prompt:
        return False
    lowered = prompt.lower()
    keywords = [
        'adata', 'qc', 'quality', 'cluster', 'clustering', 'umap', 'tsne', 'pca',
        'embedding', 'neighbors', 'leiden', 'louvain', 'marker', 'differential',
        'hvg', 'highly variable', 'preprocess', 'normalize', 'visualize', 'plot',
        '降维', '聚类', '可视化', '差异', '标记', '质控', '预处理'
    ]
    return any(keyword in lowered for keyword in keywords)
