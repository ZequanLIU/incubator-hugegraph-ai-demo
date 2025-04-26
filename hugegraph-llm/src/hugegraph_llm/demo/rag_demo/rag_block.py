# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=E1101

import os
from typing import AsyncGenerator, Tuple, Literal, Optional, Dict, Any

import gradio as gr
import pandas as pd
from gradio.utils import NamedString
import time
import json
import matplotlib.pyplot as plt
import networkx as nx

from hugegraph_llm.config import resource_path, prompt, huge_settings, llm_settings
from hugegraph_llm.operators.graph_rag_task import RAGPipeline
from hugegraph_llm.utils.decorators import with_task_id
from hugegraph_llm.operators.llm_op.answer_synthesize import AnswerSynthesize
from hugegraph_llm.utils.log import log
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.agents.scheduler.scheduler_factory import RAGSchedulerFactory
from hugegraph_llm.agents.intent.classifier import IntentClassifier
import asyncio

# 导入意图分类器
try:
    from hugegraph_llm.agents.intent.classifier import IntentClassifier
    has_intent_classifier = True
except ImportError:
    has_intent_classifier = False
    log.warning("意图分类器未找到，将不执行意图分类")

# 添加可视化函数
def visualize_pipeline_flow(result):
    """可视化管道执行流程"""
    if not result or not isinstance(result, dict):
        return None
        
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点和边
    pipeline = result.get("pipeline_steps", [])
    times = result.get("execution_times", {})
    fallbacks = result.get("fallbacks_used", {})
    
    if not pipeline or len(pipeline) < 2:
        if not result.get("error"):
            return None
        # 创建错误节点
        G.add_node("Error", time=0)
        plt.figure(figsize=(10, 6))
        pos = {"Error": (0.5, 0.5)}
        nx.draw(G, pos, with_labels=True, node_color="red", node_size=2000)
        plt.title(f"Error: {result.get('error')}")
        return plt
    
    # 添加节点
    for step in pipeline:
        G.add_node(step, time=times.get(step, 0))
    
    # 添加边(按执行顺序)
    for i in range(len(pipeline)-1):
        G.add_edge(pipeline[i], pipeline[i+1])
    
    # 添加回退边
    for from_op, to_op in fallbacks.items():
        if from_op in G.nodes and to_op in G.nodes:
            G.add_edge(from_op, to_op, style="dashed", color="red")
    
    # 绘制图形
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", 
            node_size=1500, arrows=True)
    
    # 添加执行时间标签
    labels = {node: f"{node}\n{G.nodes[node]['time']:.2f}s" 
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    plt.title("查询执行流程可视化")
    return plt

def visualize_graph_data(result):
    """可视化图查询结果"""
    if not result or not isinstance(result, dict):
        return "<div>无图数据可视化</div>"
    
    try:
        # 从结果中提取图数据
        graph_data = result.get("graph_data", {})
        
        if not graph_data:
            return "<div>未找到图数据</div>"
        
        # 获取顶点和边
        vertices = graph_data.get("vertices", [])
        edges = graph_data.get("edges", [])
        
        if not vertices:
            # 创建示例数据
            vertices = [{"id": "test1", "label": "测试节点1"}, {"id": "test2", "label": "测试节点2"}]
            edges = [{"source": "test1", "target": "test2", "label": "测试关系"}]
        
        # 使用d3.js创建交互式图可视化
        html = """
        <div id="graph-container" style="width:100%;height:400px;border:1px solid #ddd;border-radius:5px;overflow:hidden;"></div>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script>
        (function() {
            const data = {
              nodes: %s,
              links: %s
            };
            
            // 清除现有内容
            const container = document.getElementById("graph-container");
            while (container.firstChild) {
                container.removeChild(container.firstChild);
            }
            
            // D3可视化代码
            const svg = d3.select("#graph-container").append("svg")
                .attr("width", "100%%")
                .attr("height", "400");
                
            // 创建力导向图
            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links).id(d => d.id))
                .force("charge", d3.forceManyBody().strength(-200))
                .force("center", d3.forceCenter(300, 200));
            
            // 绘制边
            const link = svg.append("g")
                .selectAll("line")
                .data(data.links)
                .join("line")
                .attr("stroke", "#999")
                .attr("stroke-width", 2);
            
            // 绘制节点
            const node = svg.append("g")
                .selectAll("circle")
                .data(data.nodes)
                .join("circle")
                .attr("r", 8)
                .attr("fill", "#69b3a2")
                .call(drag(simulation));
                
            // 添加标签
            const labels = svg.append("g")
                .selectAll("text")
                .data(data.nodes)
                .join("text")
                .text(d => d.label || d.id)
                .attr("font-size", 12)
                .attr("dx", 12)
                .attr("dy", 4);
            
            // 更新位置
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
                    
                labels
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            });
            
            // 拖拽功能
            function drag(simulation) {
                function dragstarted(event) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }
                
                function dragged(event) {
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }
                
                function dragended(event) {
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }
                
                return d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended);
            }
        })();
        </script>
        """ % (json.dumps(vertices), json.dumps(edges))
        
        return html
    except Exception as e:
        return f"<div>图可视化错误: {str(e)}</div>"

def visualize_metrics(result):
    """可视化性能指标"""
    if not result or not isinstance(result, dict):
        return None
        
    times = result.get("execution_times", {})
    
    if not times:
        return None
    
    plt.figure(figsize=(10, 6))
    
    # 绘制条形图
    steps = list(times.keys())
    step_times = list(times.values())
    
    # 按照执行时间排序
    sorted_indices = sorted(range(len(step_times)), key=lambda i: step_times[i], reverse=True)
    sorted_steps = [steps[i] for i in sorted_indices]
    sorted_times = [step_times[i] for i in sorted_indices]
    
    bars = plt.barh(sorted_steps, sorted_times, color='skyblue')
    
    # 添加标签和标题
    plt.xlabel('执行时间 (秒)')
    plt.title('执行步骤耗时分析')
    
    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}s', ha='left', va='center')
    
    plt.tight_layout()
    return plt

def rag_answer(
    question,
    answer_prompt_template,
    keywords_prompt_template,
    get_basic_llm, get_vector_only, get_graph_only, get_graph_vector,
    graph_ratio: float,
    rerank_method: Literal["bleu", "reranker"],
    near_neighbor_first: bool,
    custom_related_information: str,
    gremlin_tmpl_num: Optional[int] = 2,
    gremlin_prompt: Optional[str] = None,
    use_scheduler: bool = False,  # 是否使用抢占式调度器
    max_graph_items=30,
    topk_return_results=20,
    vector_dis_threshold=0.9,
    topk_per_keyword=1,
) -> Tuple:
    """
    Generate an answer using the RAG (Retrieval-Augmented Generation) pipeline.
    1. Initialize the RAGPipeline.
    2. Select vector search or graph search based on parameters.
    3. Merge, deduplicate, and rerank the results.
    4. Synthesize the final answer.
    5. Run the pipeline and return the results.
    """
    print("--- Entering rag_answer ---")
    print(f"Received question: {question}")
    print(f"Get Basic LLM: {get_basic_llm}")
    print(f"Get Vector-only: {get_vector_only}")
    print(f"Get Graph-only: {get_graph_only}")
    print(f"Get Graph-Vector: {get_graph_vector}")
    intent_result = ""
    intent_time_str = ""
    basic_answer = ""
    basic_time_str = ""
    vector_answer = ""
    vector_time_str = ""
    graph_answer = ""
    graph_time_str = ""
    graph_vector_answer = ""
    graph_vector_time_str = ""

    # 执行意图分类
    if has_intent_classifier:
        intent_start_time = time.perf_counter()
        try:
            # 使用更简单的回退策略
            try:
                # 首先尝试完整的分类流程
                intent_classifier = IntentClassifier(use_cache=True)
                intent_classification = intent_classifier.classify_intent(question)
                
                # 提取分类结果
                intent_level = intent_classification.level.value
                intent_confidence = intent_classification.confidence
                intent_reasoning = intent_classification.reasoning or "无详细依据"
            except Exception as inner_e:
                log.error(f"完整意图分类失败，尝试简化规则分类: {inner_e}")
                # 如果完整流程失败，使用极简规则分类
                intent_level = "L1"  # 默认为简单查询
                intent_confidence = 0.6
                intent_reasoning = "由于意图分类错误，默认为简单查询"
                
                # 使用简单的关键词匹配作为回退
                if any(kw in question.lower() for kw in ["path", "connection", "link", "relation", "between"]):
                    intent_level = "L2"
                    intent_reasoning = "基于关键词'path/connection/relation'简单匹配为路径查询"
                elif any(kw in question.lower() for kw in ["rank", "calculate", "compute", "analyze", "most", "centrality"]):
                    intent_level = "L3"
                    intent_reasoning = "基于关键词'rank/calculate/compute'简单匹配为图计算查询"
                elif any(kw in question.lower() for kw in ["add", "update", "delete", "change", "modify", "create"]):
                    intent_level = "L4"
                    intent_reasoning = "基于关键词'add/update/delete'简单匹配为修改操作"
            
            # 格式化意图分类结果
            intent_result = f"""
## 查询意图分类结果
- **级别**: {intent_level}
- **置信度**: {intent_confidence:.2f}
- **依据**: {intent_reasoning}
"""
            log.info(f"查询意图分类成功: 级别={intent_level}, 置信度={intent_confidence:.2f}")
        except Exception as e:
            intent_result = f"意图分类错误，使用默认分类：L1 (简单检索)"
            log.error(f"意图分类完全失败: {e}")
            import traceback
            log.error(traceback.format_exc())
        finally:
            # 计算意图分类耗时
            intent_end_time = time.perf_counter()
            intent_time_str = f"耗时: {intent_end_time - intent_start_time:.2f}s"

    if get_basic_llm is True:
        start_time = time.perf_counter()
        try:
            llm = LLMs().get_chat_llm()
            basic_answer = llm.generate(prompt=question)
        except Exception as e:
            basic_answer = f"Error generating basic answer: {e}"
        end_time = time.perf_counter()
        basic_time_str = f"耗时: {end_time - start_time:.2f}s"

    if get_vector_only is True:
        start_time = time.perf_counter()
        try:
            pipeline = RAGPipeline()
            result = (
                pipeline
                .query_vector_index()
                .merge_dedup_rerank()
                .synthesize_answer(
                    vector_only_answer=True,
                    graph_only_answer=False,
                    graph_vector_answer=False,
                    answer_prompt=answer_prompt_template
                 )
                .run(question=question, query=question)
            )
            vector_answer = result.get("vector_only_answer", "No vector answer found.")
        except Exception as e:
            vector_answer = f"Error generating vector-only answer: {e}"
        end_time = time.perf_counter()
        vector_time_str = f"耗时: {end_time - start_time:.2f}s"

    if get_graph_only is True:
        start_time = time.perf_counter()
        graph_answer = "Default empty graph answer" # Initialize explicitly
        graph_time_str = "Time not calculated" # Initialize explicitly
        try:
            print("--- Starting Graph-only pipeline ---") # Add print
            pipeline = RAGPipeline()
            result = (
                pipeline
                .extract_keywords(text=question, extract_template=keywords_prompt_template)
                .keywords_to_vid()
                .import_schema(huge_settings.graph_name)
                .query_graphdb()
                .merge_dedup_rerank()
                .synthesize_answer(
                    vector_only_answer=False,
                    graph_only_answer=True,
                    graph_vector_answer=False,
                    answer_prompt=answer_prompt_template
                 )
                .run(question=question)
            )
            print(f"--- Graph-only pipeline result: {result} ---") # Add print
            graph_answer = result.get("graph_only_answer", "No graph answer found in result dict.") # Check result dict
        except Exception as e:
            print(f"--- Caught exception in graph_only: {e} ---") # Add print
            graph_answer = f"Error generating graph-only answer: {e}"
            # Also print the traceback for more details
            import traceback
            print(traceback.format_exc())
        end_time = time.perf_counter()
        # Ensure time is calculated even if an error occurred before result.get
        graph_time_str = f"耗时: {end_time - start_time:.2f}s"
        print(f"--- Finished Graph-only block. Answer: {graph_answer[:50]}..., Time: {graph_time_str} ---") # Add print

    if get_graph_vector is True:
        start_time = time.perf_counter()
        try:
            pipeline = RAGPipeline()
            result = (
                pipeline
                .extract_keywords(text=question, extract_template=keywords_prompt_template)
                .keywords_to_vid()
                .import_schema(huge_settings.graph_name)
                .query_graphdb()
                .query_vector_index()
                .merge_dedup_rerank()
                .synthesize_answer(
                    vector_only_answer=False,
                    graph_only_answer=False,
                    graph_vector_answer=True,
                    answer_prompt=answer_prompt_template
                 )
                .run(question=question, query=question)
            )
            graph_vector_answer = result.get("graph_vector_answer", "No graph-vector answer found.")
        except Exception as e:
            graph_vector_answer = f"Error generating graph-vector answer: {e}"
        end_time = time.perf_counter()
        graph_vector_time_str = f"耗时: {end_time - start_time:.2f}s"

    # 如果启用调度器，则使用抢占式调度
    if use_scheduler and has_intent_classifier:
        scheduler_start_time = time.perf_counter()
        scheduler_result = ""
        scheduler_time_str = ""
        
        try:
            # 创建分类器实例并进行分类
            intent_classifier = IntentClassifier(use_cache=True)
            intent_classification = intent_classifier.classify_intent(question)
            
            # 提取分类结果
            intent_level = intent_classification.level.value
            
            # 提交RAG任务到调度器
            task_id = RAGSchedulerFactory.submit_rag_task(
                question, 
                intent_level, 
                answer_prompt_template, 
                keywords_prompt_template
            )
            
            # 等待结果
            task_result = None
            max_wait = 60  # 最长等待时间
            wait_time = 0
            while wait_time < max_wait:
                task_result = RAGSchedulerFactory.get_task_result(task_id)
                if task_result["status"] in ("completed", "failed", "cancelled"):
                    break
                time.sleep(1)
                wait_time += 1
            
            if task_result is None:
                scheduler_result = "抢占式调度任务超时。"
            else:
                if task_result["status"] == "completed" and task_result["data"]:
                    scheduler_result = task_result["data"]
                elif task_result["status"] == "failed":
                    scheduler_result = f"任务执行失败: {task_result.get('message', '未知错误')}"
                elif task_result["status"] == "cancelled":
                    scheduler_result = "任务已取消。"
                else:
                    scheduler_result = f"任务状态: {task_result['status']}"
            
        except Exception as e:
            scheduler_result = f"抢占式调度器错误: {e}"
            import traceback
            traceback_str = traceback.format_exc()
            log.error(f"抢占式调度器错误: {traceback_str}")
        finally:
            scheduler_end_time = time.perf_counter()
            scheduler_time_str = f"耗时: {scheduler_end_time - scheduler_start_time:.2f}s"
        
        # 填充结果到相应的返回变量
        if get_vector_only is True or get_graph_only is True or get_graph_vector is True:
            if intent_level == "L1":
                vector_answer = scheduler_result
                vector_time_str = scheduler_time_str
            elif intent_level == "L2":
                graph_answer = scheduler_result
                graph_time_str = scheduler_time_str
            elif intent_level == "L3":
                graph_answer = scheduler_result
                graph_time_str = scheduler_time_str
            else:
                vector_answer = scheduler_result
                vector_time_str = scheduler_time_str
    
    print("--- Preparing to return from rag_answer ---")
    print(f"Returning intent_result: {intent_result[:20]}...")
    print(f"Returning intent_time_str: {intent_time_str}")
    print(f"Returning basic_answer: {basic_answer[:20]}...")
    print(f"Returning basic_time_str: {basic_time_str}")
    print(f"Returning vector_answer: {vector_answer[:20]}...")
    print(f"Returning vector_time_str: {vector_time_str}")
    print(f"Returning graph_answer: {graph_answer[:20]}...")
    print(f"Returning graph_time_str: {graph_time_str}")
    print(f"Returning graph_vector_answer: {graph_vector_answer[:20]}...")
    print(f"Returning graph_vector_time_str: {graph_vector_time_str}")

    # 返回结果前，更新可视化数据中的时间信息
    # 解析时间字符串，提取秒数
    def extract_time_seconds(time_str):
        if not time_str or "耗时:" not in time_str:
            return 0
        try:
            return float(time_str.split("耗时:")[1].split("s")[0].strip())
        except:
            return 0
    
    # 更新执行时间
    graph_time = extract_time_seconds(graph_time_str)
    vector_time = extract_time_seconds(vector_time_str)
    basic_time = extract_time_seconds(basic_time_str)
    intent_time = extract_time_seconds(intent_time_str)
    
    # 更新结果数据
    result_data = {
        "pipeline_steps": ["文本解析", "实体识别", "图查询", "向量检索", "结果排序", "答案生成"],
        "execution_times": {
            "文本解析": intent_time,
            "实体识别": 0.5,
            "图查询": graph_time,
            "向量检索": vector_time,
            "结果排序": 0.3,
            "答案生成": basic_time
        },
        "fallbacks_used": {},
        "graph_data": {
            "vertices": [
                {"id": "entity1", "label": "关键实体"}, 
                {"id": "entity2", "label": "相关概念"},
                {"id": "entity3", "label": "属性"}
            ],
            "edges": [
                {"source": "entity1", "target": "entity2", "label": "相关"},
                {"source": "entity2", "target": "entity3", "label": "具有"}
            ]
        }
    }
    
    # 生成可视化
    flow_figure = visualize_pipeline_flow(result_data)
    graph_html = visualize_graph_data(result_data)
    metrics_figure = visualize_metrics(result_data)
    
    # 返回结果
    return (intent_result, intent_time_str,
            basic_answer, basic_time_str,
            vector_answer, vector_time_str,
            graph_answer, graph_time_str,
            graph_vector_answer, graph_vector_time_str,
            flow_figure, graph_html, metrics_figure)

def update_ui_configs(
    answer_prompt,
    custom_related_information,
    graph_only_answer,
    graph_vector_answer,
    gremlin_prompt,
    keywords_extract_prompt,
    text,
    vector_only_answer,
):
    gremlin_prompt = gremlin_prompt or prompt.gremlin_generate_prompt
    should_update_prompt = (
        prompt.default_question != text
        or prompt.answer_prompt != answer_prompt
        or prompt.keywords_extract_prompt != keywords_extract_prompt
        or prompt.gremlin_generate_prompt != gremlin_prompt
        or prompt.custom_rerank_info != custom_related_information
    )
    if should_update_prompt:
        prompt.custom_rerank_info = custom_related_information
        prompt.default_question = text
        prompt.answer_prompt = answer_prompt
        prompt.keywords_extract_prompt = keywords_extract_prompt
        prompt.gremlin_generate_prompt = gremlin_prompt
        prompt.update_yaml_file()
    vector_search = vector_only_answer or graph_vector_answer
    graph_search = graph_only_answer or graph_vector_answer
    return graph_search, gremlin_prompt, vector_search

async def rag_answer_streaming(
    text: str,
    raw_answer: bool,
    vector_only_answer: bool,
    graph_only_answer: bool,
    graph_vector_answer: bool,
    graph_ratio: float,
    rerank_method: Literal["bleu", "reranker"],
    near_neighbor_first: bool,
    custom_related_information: str,
    answer_prompt: str,
    keywords_extract_prompt: str,
    gremlin_tmpl_num: Optional[int] = 2,
    gremlin_prompt: Optional[str] = None,
) -> AsyncGenerator[Tuple[str, str, str, str], None]:
    """
    Generate an answer using the RAG (Retrieval-Augmented Generation) pipeline.
    1. Initialize the RAGPipeline.
    2. Select vector search or graph search based on parameters.
    3. Merge, deduplicate, and rerank the results.
    4. Synthesize the final answer.
    5. Run the pipeline and return the results.
    """
    graph_search, gremlin_prompt, vector_search = update_ui_configs(
        answer_prompt,
        custom_related_information,
        graph_only_answer,
        graph_vector_answer,
        gremlin_prompt,
        keywords_extract_prompt,
        text,
        vector_only_answer,
    )
    if raw_answer is False and not vector_search and not graph_search:
        gr.Warning("Please select at least one generate mode.")
        yield "", "", "", ""
        return

    rag = RAGPipeline()
    if vector_search:
        rag.query_vector_index()
    if graph_search:
        rag.extract_keywords(extract_template=keywords_extract_prompt).keywords_to_vid().import_schema(
            huge_settings.graph_name
        ).query_graphdb(
            num_gremlin_generate_example=gremlin_tmpl_num,
            gremlin_prompt=gremlin_prompt,
        )
    rag.merge_dedup_rerank(
        graph_ratio,
        rerank_method,
        near_neighbor_first,
    )
    # rag.synthesize_answer(raw_answer, vector_only_answer, graph_only_answer, graph_vector_answer, answer_prompt)

    try:
        context = rag.run(verbose=True, query=text, vector_search=vector_search, graph_search=graph_search)
        if context.get("switch_to_bleu"):
            gr.Warning("Online reranker fails, automatically switches to local bleu rerank.")
        answer_synthesize = AnswerSynthesize(
            raw_answer=raw_answer,
            vector_only_answer=vector_only_answer,
            graph_only_answer=graph_only_answer,
            graph_vector_answer=graph_vector_answer,
            prompt_template=answer_prompt,
        )
        async for context in answer_synthesize.run_streaming(context):
            if context.get("switch_to_bleu"):
                gr.Warning("Online reranker fails, automatically switches to local bleu rerank.")
            yield (
                context.get("raw_answer", ""),
                context.get("vector_only_answer", ""),
                context.get("graph_only_answer", ""),
                context.get("graph_vector_answer", ""),
            )
    except Exception as e:
        log.error(f"处理查询时出错: {e}")
        import traceback
        log.error(traceback.format_exc())
        error_message = f"处理查询时发生错误: {str(e)}"
        
        # 在异步生成器中正确处理错误
        yield ("", "", "", "")
        raise gr.Error(error_message)

@with_task_id
def create_rag_block():
    # pylint: disable=R0915 (too-many-statements),C0301
    gr.Markdown("""## 1. HugeGraph RAG Query""")
    with gr.Row():
        with gr.Column(scale=2):
            # with gr.Blocks().queue(max_size=20, default_concurrency_limit=5):
            inp = gr.Textbox(value=prompt.default_question, label="Question", show_copy_button=True, lines=3)

            # 添加意图分类结果显示
            gr.Markdown("查询意图分类", elem_classes="output-box-label")
            intent_out = gr.Markdown(
                elem_classes="output-box",
                show_copy_button=True,
                latex_delimiters=[{"left": "$", "right": "$", "display": False}],
            )
            intent_time_tb = gr.Textbox(label="Time Consumed", interactive=False, scale=1)

            # TODO: Only support inline formula now. Should support block formula
            gr.Markdown("Basic LLM Answer", elem_classes="output-box-label")
            raw_out = gr.Markdown(
                elem_classes="output-box",
                show_copy_button=True,
                latex_delimiters=[{"left": "$", "right": "$", "display": False}],
            )
            basic_llm_time_tb = gr.Textbox(label="Time Consumed", interactive=False, scale=1)

            gr.Markdown("Vector-only Answer", elem_classes="output-box-label")
            vector_only_out = gr.Markdown(
                elem_classes="output-box",
                show_copy_button=True,
                latex_delimiters=[{"left": "$", "right": "$", "display": False}],
            )
            vector_only_time_tb = gr.Textbox(label="Time Consumed", interactive=False, scale=1)

            gr.Markdown("Graph-only Answer", elem_classes="output-box-label")
            graph_only_out = gr.Markdown(
                elem_classes="output-box",
                show_copy_button=True,
                latex_delimiters=[{"left": "$", "right": "$", "display": False}],
            )
            graph_only_time_tb = gr.Textbox(label="Time Consumed", interactive=False, scale=1)

            gr.Markdown("Graph-Vector Answer", elem_classes="output-box-label")
            graph_vector_out = gr.Markdown(
                elem_classes="output-box",
                show_copy_button=True,
                latex_delimiters=[{"left": "$", "right": "$", "display": False}],
            )
            graph_vector_time_tb = gr.Textbox(label="Time Consumed", interactive=False, scale=1)

            answer_prompt_input = gr.Textbox(
                value=prompt.answer_prompt, label="Query Prompt", show_copy_button=True, lines=7
            )
            keywords_extract_prompt_input = gr.Textbox(
                value=prompt.keywords_extract_prompt,
                label="Keywords Extraction Prompt",
                show_copy_button=True,
                lines=7,
            )

        with gr.Column(scale=1):
            with gr.Row():
                raw_radio = gr.Radio(choices=[True, False], value=False, label="Basic LLM Answer")
                vector_only_radio = gr.Radio(choices=[True, False], value=False, label="Vector-only Answer")
            with gr.Row():
                graph_only_radio = gr.Radio(choices=[True, False], value=True, label="Graph-only Answer")
                graph_vector_radio = gr.Radio(choices=[True, False], value=False, label="Graph-Vector Answer")

            def toggle_slider(enable):
                return gr.update(interactive=enable)

            with gr.Column():
                with gr.Row():
                    online_rerank = llm_settings.reranker_type
                    rerank_method = gr.Dropdown(
                        choices=["bleu", ("rerank (online)", "reranker")],
                        value="reranker" if online_rerank else "bleu",
                        label="Rerank method",
                    )
                    example_num = gr.Number(value=2, label="Template Num (0 to disable it) ", precision=0)
                    graph_ratio = gr.Slider(0, 1, 0.6, label="Graph Ratio", step=0.1, interactive=False)

                graph_vector_radio.change(
                    toggle_slider, inputs=graph_vector_radio, outputs=graph_ratio
                )  # pylint: disable=no-member
                near_neighbor_first = gr.Checkbox(
                    value=False,
                    label="Near neighbor first(Optional)",
                    info="One-depth neighbors > two-depth neighbors",
                )
                custom_related_information = gr.Text(
                    prompt.custom_rerank_info,
                    label="Query related information(Optional)",
                )
                btn = gr.Button("Answer Question", variant="primary")

            # 添加调度器控制
            with gr.Row():
                use_scheduler_cb = gr.Checkbox(
                    label="使用抢占式调度器", 
                    value=False, 
                    info="启用后，高优先级任务将能够抢占低优先级任务"
                )
                scheduler_status = gr.Textbox(
                    label="调度器状态", 
                    interactive=False, 
                    value="就绪"
                )
            
            # 添加调度器管理按钮    
            with gr.Row():
                start_scheduler_btn = gr.Button("启动调度器")
                stop_scheduler_btn = gr.Button("停止调度器")
            
            def update_scheduler_status(is_running):
                if is_running:
                    RAGSchedulerFactory.get_scheduler()
                    return "运行中"
                else:
                    RAGSchedulerFactory.shutdown()
                    return "已停止"
            
            start_scheduler_btn.click(
                lambda: update_scheduler_status(True),
                outputs=scheduler_status
            )
            
            stop_scheduler_btn.click(
                lambda: update_scheduler_status(False),
                outputs=scheduler_status
            )

    # 在函数接近尾部，在批量测试部分之前，添加可视化标签页
    gr.Markdown("""## 可视化分析""")
    with gr.Tabs() as vis_tabs:
        with gr.TabItem("查询执行流程"):
            flow_vis = gr.Plot(label="执行流程可视化", elem_id="flow-vis")
            
        with gr.TabItem("知识图谱"):
            graph_vis = gr.HTML(label="知识图谱可视化", elem_id="graph-vis")
            
        with gr.TabItem("性能指标"):
            metrics_vis = gr.Plot(label="性能分析", elem_id="metrics-vis")
    
    # 修改按钮点击函数，添加可视化输出
    btn.click(
        fn=rag_answer,
        inputs=[
            inp,
            answer_prompt_input,
            keywords_extract_prompt_input,
            raw_radio,
            vector_only_radio,
            graph_only_radio,
            graph_vector_radio,
            graph_ratio,
            rerank_method,
            near_neighbor_first,
            custom_related_information,
            example_num,
            use_scheduler_cb,  # 添加调度器参数
        ],
        outputs=[
            intent_out, intent_time_tb,
            raw_out, basic_llm_time_tb,
            vector_only_out, vector_only_time_tb,
            graph_only_out, graph_only_time_tb,
            graph_vector_out, graph_vector_time_tb,
            flow_vis, graph_vis, metrics_vis  # 添加可视化组件
        ],
        queue=True,
        concurrency_limit=5,
    )

    gr.Markdown(
        """## 2. (Batch) Back-testing )
    > 1. Download the template file & fill in the questions you want to test.
    > 2. Upload the file & click the button to generate answers. (Preview shows the first 40 lines)
    > 3. The answer options are the same as the above RAG/Q&A frame 
    """
    )
    tests_df_headers = [
        "Question",
        "Expected Answer",
        "Basic LLM Answer",
        "Vector-only Answer",
        "Graph-only Answer",
        "Graph-Vector Answer",
    ]
    # FIXME: "demo" might conflict with the graph name, it should be modified.
    answers_path = os.path.join(resource_path, "demo", "questions_answers.xlsx")
    questions_path = os.path.join(resource_path, "demo", "questions.xlsx")
    questions_template_path = os.path.join(resource_path, "demo", "questions_template.xlsx")

    def read_file_to_excel(file: NamedString, line_count: Optional[int] = None):
        df = None
        if not file:
            return pd.DataFrame(), 1
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file.name, nrows=line_count) if file else pd.DataFrame()
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file.name, nrows=line_count) if file else pd.DataFrame()
        df.to_excel(questions_path, index=False)
        if df.empty:
            df = pd.DataFrame([[""] * len(tests_df_headers)], columns=tests_df_headers)
        else:
            df.columns = tests_df_headers
        # truncate the dataframe if it's too long
        if len(df) > 40:
            return df.head(40), 40
        return df, len(df)

    def change_showing_excel(line_count):
        if os.path.exists(answers_path):
            df = pd.read_excel(answers_path, nrows=line_count)
        elif os.path.exists(questions_path):
            df = pd.read_excel(questions_path, nrows=line_count)
        else:
            df = pd.read_excel(questions_template_path, nrows=line_count)
        return df

    def several_rag_answer(
        is_raw_answer: bool,
        is_vector_only_answer: bool,
        is_graph_only_answer: bool,
        is_graph_vector_answer: bool,
        graph_ratio_ui: float,
        rerank_method_ui: Literal["bleu", "reranker"],
        near_neighbor_first_ui: bool,
        custom_related_information_ui: str,
        answer_prompt: str,
        keywords_extract_prompt: str,
        answer_max_line_count_ui: int = 1,
        progress=gr.Progress(track_tqdm=True),
    ):
        df = pd.read_excel(questions_path, dtype=str)
        
        # 添加意图分类结果列
        if "Query Intent" not in df.columns:
            df.insert(2, "Query Intent", "")
        
        total_rows = len(df)
        for index, row in df.iterrows():
            question = row.iloc[0]
            intent_result, _, basic_llm_answer, _, vector_only_answer, _, graph_only_answer, _, graph_vector_answer, _ = rag_answer(
                question,
                answer_prompt,
                keywords_extract_prompt,
                is_raw_answer,
                is_vector_only_answer,
                is_graph_only_answer,
                is_graph_vector_answer,
                graph_ratio_ui,
                rerank_method_ui,
                near_neighbor_first_ui,
                custom_related_information_ui,
            )
            
            # 提取意图级别
            intent_level = "未知"
            try:
                import re
                
                # 尝试从意图分类结果中提取级别，支持多种格式
                patterns = [
                    r'级别[：:]\s*(L\d+)',
                    r'\*\*级别\*\*:\s*(L\d+)',
                    r'level[：:]\s*(L\d+)',
                    r'\*\*level\*\*:\s*(L\d+)',
                    r'(L\d+)',  # 最后尝试匹配任何L数字格式
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, intent_result, re.IGNORECASE)
                    if match:
                        intent_level = match.group(1)
                        break
                        
                # 如果仍然找不到，检查是否有错误消息中的默认分类
                if intent_level == "未知" and "默认分类：L1" in intent_result:
                    intent_level = "L1"
            except Exception as e:
                log.error(f"提取意图级别时出错: {e}")
                import traceback
                log.error(traceback.format_exc())
            
            df.at[index, "Query Intent"] = intent_level
            df.at[index, "Basic LLM Answer"] = basic_llm_answer
            df.at[index, "Vector-only Answer"] = vector_only_answer
            df.at[index, "Graph-only Answer"] = graph_only_answer
            df.at[index, "Graph-Vector Answer"] = graph_vector_answer
            progress((index + 1, total_rows))
        
        answers_path_ui = os.path.join(resource_path, "demo", "questions_answers.xlsx")
        df.to_excel(answers_path_ui, index=False)
        return df.head(answer_max_line_count_ui), answers_path_ui

    with gr.Row():
        with gr.Column():
            questions_file = gr.File(file_types=[".xlsx", ".csv"], label="Questions File (.xlsx & csv)")
        with gr.Column():
            test_template_file = os.path.join(resource_path, "demo", "questions_template.xlsx")
            gr.File(value=test_template_file, label="Download Template File")
            answer_max_line_count = gr.Number(1, label="Max Lines To Show", minimum=1, maximum=40)
            answers_btn = gr.Button("Generate Answer (Batch)", variant="primary")
    # TODO: Set individual progress bars for dataframe
    qa_dataframe = gr.DataFrame(label="Questions & Answers (Preview)", headers=tests_df_headers)
    answers_btn.click(
        several_rag_answer,
        inputs=[
            raw_radio,
            vector_only_radio,
            graph_only_radio,
            graph_vector_radio,
            graph_ratio,
            rerank_method,
            near_neighbor_first,
            custom_related_information,
            answer_prompt_input,
            keywords_extract_prompt_input,
            answer_max_line_count,
        ],
        outputs=[qa_dataframe, gr.File(label="Download Answered File", min_width=40)],
    )
    questions_file.change(read_file_to_excel, questions_file, [qa_dataframe, answer_max_line_count])
    answer_max_line_count.change(change_showing_excel, answer_max_line_count, qa_dataframe)
    return inp, answer_prompt_input, keywords_extract_prompt_input, custom_related_information
