"""
可组合RAG示例应用
展示如何在Gradio界面中使用可组合RAG管道
"""
import os
import time
import json
import gradio as gr
import logging
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, List, Tuple

from hugegraph_llm.rag.pipeline import PipelineBuilder
from hugegraph_llm.rag.memory import DialogueMemory
from hugegraph_llm.rag.degradation import get_degradation_manager
from hugegraph_llm.llm import LLMs
from hugegraph_llm.utils.hugegraph_client import HugeGraphClient
from hugegraph_llm.config import huge_settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建服务
llm_service = LLMs().get_chat_llm()
graph_client = HugeGraphClient(huge_settings.server_url)

# 创建降级管理器
degradation_manager = get_degradation_manager()

# 创建内存
session_memory = DialogueMemory()

# 构建RAG管道
def build_rag_pipeline():
    """构建RAG管道"""
    builder = PipelineBuilder("HugeGraph可组合RAG")
    
    # 添加服务
    builder.with_service("llm_service", llm_service)
    builder.with_service("graph_client", graph_client)
    builder.with_service("fallback_model", "lightweight")
    
    # 添加微操作
    builder.add_entity_recognition()
    builder.add_graph_query(with_fallback=True)
    builder.add_result_refinement(with_fallback=True)
    
    return builder.build()

# 全局管道实例
rag_pipeline = build_rag_pipeline()

def visualize_pipeline_flow(result):
    """可视化管道执行流程"""
    if not result or not isinstance(result, dict):
        return None
        
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点和边
    stats = result.get("execution_stats", {})
    operations = list(stats.get("operation_times", {}).keys())
    fallbacks = stats.get("fallbacks_used", {})
    
    if not operations:
        return None
    
    # 添加节点
    for op in operations:
        G.add_node(op, time=stats["operation_times"].get(op, 0))
    
    # 添加边(按执行顺序)
    for i in range(len(operations)-1):
        G.add_edge(operations[i], operations[i+1])
    
    # 添加回退边
    for op, fallback in fallbacks.items():
        if op in G.nodes and fallback in G.nodes:
            G.add_edge(op, fallback, style="dashed", color="red")
    
    # 绘制图形
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", 
            node_size=1500, arrows=True, ax=ax)
    
    # 添加执行时间标签
    labels = {node: f"{node}\n{G.nodes[node]['time']:.2f}s" 
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    plt.title("执行流程")
    return fig

def visualize_graph_data(result):
    """可视化图查询结果"""
    if not result or not isinstance(result, dict) or "GraphQuery_result" not in result:
        return "<div>无图数据可视化</div>"
    
    try:
        graph_data = result["GraphQuery_result"]
        
        # 获取顶点和边
        # 注意：这里需要根据实际的图数据结构进行调整
        if not hasattr(graph_data, 'vertices') or not hasattr(graph_data, 'edges'):
            # 尝试从结果中提取顶点和边
            vertices = []
            edges = []
            
            if hasattr(graph_data, 'data'):
                # 假设数据格式是 {data: {vertices: [...], edges: [...]}}
                if hasattr(graph_data.data, 'vertices'):
                    vertices = graph_data.data.vertices
                if hasattr(graph_data.data, 'edges'):
                    edges = graph_data.data.edges
            
            # 如果没有找到数据，创建示例数据用于测试
            if not vertices:
                vertices = [{"id": "test1", "label": "测试节点1"}, {"id": "test2", "label": "测试节点2"}]
                edges = [{"source": "test1", "target": "test2", "label": "测试关系"}]
        else:
            vertices = graph_data.vertices
            edges = graph_data.edges
        
        # 转换为JSON
        vertices_json = json.dumps(vertices)
        edges_json = json.dumps(edges)
        
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
        """ % (vertices_json, edges_json)
        
        return html
    except Exception as e:
        logger.exception(f"生成图可视化时出错: {e}")
        return f"<div>生成图可视化时出错: {str(e)}</div>"

def visualize_metrics(result):
    """可视化性能指标"""
    if not result or not isinstance(result, dict):
        return None
        
    stats = result.get("execution_stats", {})
    operation_times = stats.get("operation_times", {})
    
    if not operation_times:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制条形图
    operations = list(operation_times.keys())
    times = list(operation_times.values())
    
    # 按照执行时间排序
    sorted_indices = sorted(range(len(times)), key=lambda i: times[i], reverse=True)
    sorted_ops = [operations[i] for i in sorted_indices]
    sorted_times = [times[i] for i in sorted_indices]
    
    bars = ax.barh(sorted_ops, sorted_times, color='skyblue')
    
    # 添加标签和标题
    ax.set_xlabel('执行时间 (秒)')
    ax.set_title('微操作执行时间')
    
    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}s', ha='left', va='center')
    
    plt.tight_layout()
    return fig

def process_query(
    query: str,
    show_stats: bool = False
) -> Tuple[str, str, str, str, Any, Any, Any]:
    """处理用户查询"""
    start_time = time.perf_counter()
    
    try:
        # 添加到内存
        session_memory.add_user_message(query)
        
        # 准备上下文
        context = {
            "query": query,
            "memory": session_memory.build_memory_context(),
            "timestamp": time.time()
        }
        
        # 执行管道
        result = rag_pipeline.execute(context)
        
        # 提取答案
        answer = "无法生成答案"
        if "ResultRefinement_result" in result:
            answer = result["ResultRefinement_result"].answer
        
        # 添加到内存
        session_memory.add_system_message(answer)
        
        # 计算总时间
        total_time = time.perf_counter() - start_time
        
        # 准备统计信息
        stats = result.get("execution_stats", {})
        operation_times = stats.get("operation_times", {})
        fallbacks = stats.get("fallbacks_used", {})
        degradations = stats.get("degradations", [])
        
        # 格式化统计信息
        stats_text = ""
        if show_stats:
            stats_text = f"### 执行统计\n"
            stats_text += f"总执行时间: {total_time:.2f}s\n\n"
            
            stats_text += "#### 微操作耗时\n"
            for op, op_time in operation_times.items():
                stats_text += f"- {op}: {op_time:.2f}s\n"
            
            if fallbacks:
                stats_text += "\n#### 使用的回退策略\n"
                for op, fallback in fallbacks.items():
                    stats_text += f"- {op} → {fallback}\n"
            
            if degradations:
                stats_text += "\n#### 系统降级\n"
                for deg in degradations:
                    stats_text += f"- {deg['operation']} ({deg['level']}): {deg['reason']}\n"
        
        # 检查系统健康状态
        health_status = "正常"
        health_details = ""
        
        for service in ["EntityRecognition", "GremlinGraphQuery", "ResultRefinement"]:
            status = degradation_manager.get_degradation_status(service)
            if status["degraded"]:
                health_status = "降级"
                level = status["level"]
                reason = status["reason"]
                health_details += f"- {service}: {level} ({reason})\n"
        
        if not health_details:
            health_details = "所有服务正常运行"
        
        # 生成可视化
        flow_figure = visualize_pipeline_flow(result) if show_stats else None
        graph_html = visualize_graph_data(result)
        metrics_figure = visualize_metrics(result) if show_stats else None
        
        # 返回结果
        return answer, stats_text, health_status, health_details, flow_figure, graph_html, metrics_figure
        
    except Exception as e:
        logger.exception(f"处理查询时出错: {e}")
        error_message = f"处理查询时发生错误: {str(e)}"
        session_memory.add_system_message(error_message)
        return error_message, "", "错误", f"发生系统错误: {str(e)}", None, None, None

def create_demo_interface():
    """创建演示界面"""
    with gr.Blocks(title="HugeGraph 可组合RAG演示") as demo:
        gr.Markdown("## HugeGraph 可组合RAG演示")
        gr.Markdown("这个演示展示了HugeGraph的可组合RAG功能，包括自动回退和降级策略。")
        
        with gr.Tabs() as tabs:
            with gr.TabItem("查询"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="输入您的问题",
                            placeholder="例如：华为是哪一年成立的？",
                            lines=3
                        )
                        
                        with gr.Row():
                            submit_btn = gr.Button("提交", variant="primary")
                            clear_btn = gr.Button("清除")
                            show_stats_cb = gr.Checkbox(label="显示执行统计", value=True)
                        
                        answer_output = gr.Markdown(label="回答")
                        stats_output = gr.Markdown(label="执行统计")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 系统状态")
                        health_status = gr.Textbox(label="健康状态", interactive=False)
                        health_details = gr.Markdown(label="详细信息")
                        
                        with gr.Row():
                            reset_btn = gr.Button("重置降级状态")
                        
                        gr.Markdown("### 测试命令")
                        with gr.Row():
                            test_error_btn = gr.Button("模拟图查询错误")
                            test_llm_btn = gr.Button("模拟LLM错误")
            
            with gr.TabItem("可视化"):
                with gr.Row():
                    with gr.Column(scale=1):
                        flow_vis = gr.Plot(label="执行流程可视化")
                        metrics_vis = gr.Plot(label="性能指标")
                    with gr.Column(scale=1):
                        graph_vis = gr.HTML(label="知识图谱可视化")
        
        # 事件处理
        submit_btn.click(
            process_query,
            inputs=[query_input, show_stats_cb],
            outputs=[answer_output, stats_output, health_status, health_details, 
                    flow_vis, graph_vis, metrics_vis]
        )
        
        clear_btn.click(
            lambda: ("", "", "", "", None, None, None),
            inputs=[],
            outputs=[query_input, answer_output, stats_output, health_details, 
                   flow_vis, graph_vis, metrics_vis]
        )
        
        reset_btn.click(
            lambda: (degradation_manager.reset_all_degradations(), "正常", "所有服务已重置为正常状态"),
            inputs=[],
            outputs=[health_status, health_details]
        )
        
        # 测试函数
        def simulate_graph_error():
            # 模拟图查询错误
            for _ in range(3):
                degradation_manager.record_error("GremlinGraphQuery", "查询超时")
            
            status = degradation_manager.get_degradation_status("GremlinGraphQuery")
            if status["degraded"]:
                return "降级", f"图查询服务已降级: {status['level']} ({status['reason']})"
            return "正常", "图查询服务仍正常运行（需要更多错误才能触发降级）"
        
        def simulate_llm_error():
            # 模拟LLM错误
            for _ in range(5):
                degradation_manager.record_error("ResultRefinement", "LLM服务超时")
            
            status = degradation_manager.get_degradation_status("ResultRefinement")
            if status["degraded"]:
                return "降级", f"结果精炼服务已降级: {status['level']} ({status['reason']})"
            return "正常", "结果精炼服务仍正常运行（需要更多错误才能触发降级）"
        
        test_error_btn.click(
            simulate_graph_error,
            inputs=[],
            outputs=[health_status, health_details]
        )
        
        test_llm_btn.click(
            simulate_llm_error,
            inputs=[],
            outputs=[health_status, health_details]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860) 