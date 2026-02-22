import React, { useCallback, useState, useMemo } from 'react'
import {
  ReactFlow,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  NodeTypes,
  Handle,
  Position,
  ReactFlowProvider,
  useReactFlow,
  Panel,
} from 'reactflow'
import 'reactflow/dist/style.css'

const NODE_TYPES = ['agent', 'tool', 'rag', 'team', 'condition', 'output'] as const
type NodeType = typeof NODE_TYPES[number]

const NODE_SCHEMAS: Record<NodeType, { label: string; fields: { key: string; label: string; type: string }[] }> = {
  agent: { label: 'Agent', fields: [{ key: 'agent_id', label: 'Agent ID', type: 'string' }, { key: 'type', label: 'Type', type: 'select' }] },
  tool: { label: 'Tool', fields: [{ key: 'tool_name', label: 'Tool Name', type: 'string' }] },
  rag: { label: 'RAG', fields: [{ key: 'collection', label: 'Collection', type: 'string' }] },
  team: { label: 'Team', fields: [{ key: 'team_id', label: 'Team ID', type: 'string' }] },
  condition: { label: 'Condition', fields: [{ key: 'condition_expr', label: 'Expression', type: 'string' }] },
  output: { label: 'Output', fields: [{ key: 'output_key', label: 'Output Key', type: 'string' }] },
}

const STATUS_COLORS: Record<string, string> = {
  pending: '#6b7280',
  running: '#3b82f6',
  done: '#22c55e',
  error: '#ef4444',
}

function BaseNode({ data, type, status }: { data: Record<string, unknown>; type: NodeType; status?: string }) {
  const schema = NODE_SCHEMAS[type]
  const borderColor = status ? STATUS_COLORS[status] || '#374151' : '#374151'
  return (
    <div style={{ padding: 12, minWidth: 120, borderRadius: 8, border: `2px solid ${borderColor}`, background: '#1f2937' }}>
      <Handle type="target" position={Position.Top} />
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{schema.label}</div>
      <div style={{ fontSize: 12, color: '#9ca3af' }}>
        {(data.agent_id || data.tool_name || data.collection || data.team_id || data.output_key || data.condition_expr || '—') as string}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}

const nodeComponents: NodeTypes = {
  agent: (p) => <BaseNode {...p} type="agent" status={p.data.status as string} />,
  tool: (p) => <BaseNode {...p} type="tool" status={p.data.status as string} />,
  rag: (p) => <BaseNode {...p} type="rag" status={p.data.status as string} />,
  team: (p) => <BaseNode {...p} type="team" status={p.data.status as string} />,
  condition: (p) => <BaseNode {...p} type="condition" status={p.data.status as string} />,
  output: (p) => <BaseNode {...p} type="output" status={p.data.status as string} />,
}

function serialize(nodes: Node[], edges: Edge[]): string {
  const dagNodes = nodes.map((n) => {
    const d = n.data as Record<string, unknown>
    const base: Record<string, unknown> = { id: n.id }
    if (n.type === 'agent') {
      base.agent_id = d.agent_id || 'agent'
      base.type = d.type || 'sequential'
    } else if (n.type === 'tool') {
      base.agent_id = d.tool_name || 'tool'
      base.type = 'sequential'
    } else if (n.type === 'rag') {
      base.agent_id = d.collection || 'default'
      base.type = 'sequential'
    } else if (n.type === 'team') {
      base.agent_id = d.team_id || 'team'
      base.type = 'sequential'
    } else if (n.type === 'condition') {
      base.agent_id = 'condition'
      base.type = 'condition'
    } else if (n.type === 'output') {
      base.agent_id = d.output_key || 'output'
      base.type = 'sequential'
    } else {
      base.agent_id = 'node'
      base.type = 'sequential'
    }
    return base
  })
  const dagEdges = edges.map((e) => {
    const out: Record<string, string> = { source: e.source, target: e.target }
    if (e.data?.condition_expr) out.condition_expr = e.data.condition_expr
    return out
  })
  const yaml = `nodes:\n${dagNodes.map((n) => `  - id: ${n.id}\n    agent_id: ${n.agent_id}\n    type: ${n.type}`).join('\n')}\nedges:\n${dagEdges.map((e) => `  - source: ${e.source}\n    target: ${e.target}${e.condition_expr ? `\n    condition_expr: "${e.condition_expr}"` : ''}`).join('\n')}`
  return yaml
}

function VisualBuilderInner() {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [workflowId, setWorkflowId] = useState<string | null>(null)
  const [workflowName, setWorkflowName] = useState('')
  const [nodeStatus, setNodeStatus] = useState<Record<string, string>>({})
  const { project } = useReactFlow()

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge({ ...params, type: 'smoothstep' }, eds)),
    [setEdges],
  )

  const edgesWithLabels = useMemo(
    () =>
      edges.map((e) => ({
        ...e,
        label: (e.data as Record<string, unknown>)?.condition_expr as string | undefined,
        labelStyle: { fill: '#9ca3af', fontSize: 10 },
        labelBgStyle: { fill: '#1f2937' },
        labelBgPadding: [4, 2] as [number, number],
        labelBgBorderRadius: 4,
      })),
    [edges],
  )

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNode(node)
  }, [])

  const addNode = useCallback(
    (type: NodeType) => {
      const id = `${type}-${Date.now()}`
      const schema = NODE_SCHEMAS[type]
      const data: Record<string, unknown> = { status: 'pending' }
      schema.fields.forEach((f) => { data[f.key] = f.type === 'select' ? 'sequential' : '' })
      setNodes((nds) => [
        ...nds,
        {
          id,
          type,
          position: project({ x: 100 + nds.length * 50, y: 100 + nds.length * 80 }),
          data,
        },
      ])
    },
    [setNodes, project],
  )

  const updateNodeData = useCallback(
    (key: string, value: string) => {
      if (!selectedNode) return
      setNodes((nds) =>
        nds.map((n) =>
          n.id === selectedNode.id ? { ...n, data: { ...n.data, [key]: value } } : n,
        ),
      )
      setSelectedNode((prev) => (prev ? { ...prev, data: { ...prev.data, [key]: value } } : null))
    },
    [selectedNode, setNodes],
  )

  const updateEdgeCondition = useCallback(
    (edgeId: string, condition_expr: string) => {
      setEdges((eds) =>
        eds.map((e) => (e.id === edgeId ? { ...e, data: { ...e.data, condition_expr } } : e)),
      )
    },
    [setEdges],
  )

  const handleSave = useCallback(async () => {
    const name = workflowName || `workflow-${Date.now()}`
    const dag = serialize(nodes, edges)
    try {
      const res = await fetch('/workflows/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, dag }),
      })
      const data = await res.json()
      if (data.id) {
        setWorkflowId(data.id)
        setWorkflowName(name)
      }
    } catch (e) {
      console.error(e)
    }
  }, [nodes, edges, workflowName])

  const handleRun = useCallback(async () => {
    const id = workflowId
    if (!id) return
    setNodeStatus({})
    nodes.forEach((n) => setNodeStatus((s) => ({ ...s, [n.id]: 'pending' })))
    try {
      await fetch(`/workflows/${id}/run`, { method: 'POST' })
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = (window as unknown as { VITE_WS_HOST?: string }).VITE_WS_HOST || window.location.host
      const ws = new WebSocket(`${protocol}//${host}/ws/monitor`)
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          if (msg.node_id && msg.status) {
            setNodeStatus((s) => ({ ...s, [msg.node_id]: msg.status }))
          }
        } catch (_) {}
      }
    } catch (e) {
      console.error(e)
    }
  }, [workflowId, nodes])

  const nodesWithStatus = useMemo(
    () =>
      nodes.map((n) => ({
        ...n,
        data: { ...n.data, status: nodeStatus[n.id] || (n.data as Record<string, unknown>).status },
      })),
    [nodes, nodeStatus],
  )

  const selectedEdges = useMemo(
    () => edges.filter((e) => selectedNode && (e.source === selectedNode.id || e.target === selectedNode.id)),
    [edges, selectedNode],
  )

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex' }}>
      <ReactFlow
        nodes={nodesWithStatus}
        edges={edgesWithLabels}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={() => setSelectedNode(null)}
        nodeTypes={nodeComponents}
        fitView
      >
        <Background />
        <Controls />
        <Panel position="top-left" style={{ display: 'flex', gap: 8, margin: 8 }}>
          <select
            onChange={(e) => {
              const v = e.target.value
              if (v) addNode(v as NodeType)
              e.target.value = ''
            }}
            style={{ padding: 8, borderRadius: 6, background: '#1f2937', color: '#fff', border: '1px solid #374151' }}
          >
            <option value="">Add Node</option>
            {NODE_TYPES.map((t) => (
              <option key={t} value={t}>
                {NODE_SCHEMAS[t].label}
              </option>
            ))}
          </select>
          <input
            placeholder="Workflow name"
            value={workflowName}
            onChange={(e) => setWorkflowName(e.target.value)}
            style={{ padding: 8, borderRadius: 6, background: '#1f2937', color: '#fff', border: '1px solid #374151', width: 140 }}
          />
          <button
            onClick={handleSave}
            style={{ padding: '8px 16px', borderRadius: 6, background: '#3b82f6', color: '#fff', border: 'none', cursor: 'pointer' }}
          >
            Save
          </button>
          <button
            onClick={handleRun}
            disabled={!workflowId}
            style={{
              padding: '8px 16px',
              borderRadius: 6,
              background: workflowId ? '#22c55e' : '#374151',
              color: '#fff',
              border: 'none',
              cursor: workflowId ? 'pointer' : 'not-allowed',
            }}
          >
            Run
          </button>
        </Panel>
      </ReactFlow>
      {selectedNode && (
        <div
          style={{
            width: 280,
            padding: 16,
            background: '#1f2937',
            borderLeft: '1px solid #374151',
            overflowY: 'auto',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 12 }}>
            {NODE_SCHEMAS[selectedNode.type as NodeType]?.label || selectedNode.type} ({selectedNode.id})
          </div>
          {(NODE_SCHEMAS[selectedNode.type as NodeType]?.fields || []).map((f) => (
            <div key={f.key} style={{ marginBottom: 12 }}>
              <label style={{ display: 'block', fontSize: 12, color: '#9ca3af', marginBottom: 4 }}>{f.label}</label>
              {f.type === 'select' ? (
                <select
                  value={(selectedNode.data as Record<string, unknown>)[f.key] as string}
                  onChange={(e) => updateNodeData(f.key, e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 6, background: '#111827', color: '#fff', border: '1px solid #374151' }}
                >
                  <option value="sequential">sequential</option>
                  <option value="parallel">parallel</option>
                  <option value="condition">condition</option>
                </select>
              ) : (
                <input
                  value={((selectedNode.data as Record<string, unknown>)[f.key] as string) || ''}
                  onChange={(e) => updateNodeData(f.key, e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 6, background: '#111827', color: '#fff', border: '1px solid #374151' }}
                />
              )}
            </div>
          ))}
          {selectedNode.type === 'condition' && selectedEdges.filter((e) => e.source === selectedNode.id).length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 12, color: '#9ca3af', marginBottom: 8 }}>Outgoing edge conditions</div>
              {selectedEdges
                .filter((e) => e.source === selectedNode.id)
                .map((e) => (
                  <div key={e.id} style={{ marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: '#6b7280' }}>→ {e.target}</span>
                    <input
                      placeholder="condition_expr"
                      value={((e.data as Record<string, unknown>)?.condition_expr as string) || ''}
                      onChange={(ev) => updateEdgeCondition(e.id!, ev.target.value)}
                      style={{ width: '100%', padding: 6, marginTop: 4, borderRadius: 4, background: '#111827', color: '#fff', border: '1px solid #374151', fontSize: 12 }}
                    />
                  </div>
                ))}
            </div>
          )}
          {selectedEdges.filter((e) => e.target === selectedNode.id).length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 12, color: '#9ca3af', marginBottom: 8 }}>Incoming edge conditions</div>
              {selectedEdges
                .filter((e) => e.target === selectedNode.id)
                .map((e) => (
                  <div key={e.id} style={{ marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: '#6b7280' }}>{e.source} →</span>
                    <input
                      placeholder="condition_expr"
                      value={((e.data as Record<string, unknown>)?.condition_expr as string) || ''}
                      onChange={(ev) => updateEdgeCondition(e.id!, ev.target.value)}
                      style={{ width: '100%', padding: 6, marginTop: 4, borderRadius: 4, background: '#111827', color: '#fff', border: '1px solid #374151', fontSize: 12 }}
                    />
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function VisualBuilder() {
  return (
    <ReactFlowProvider>
      <div style={{ width: '100vw', height: '100vh' }}>
        <VisualBuilderInner />
      </div>
    </ReactFlowProvider>
  )
}
