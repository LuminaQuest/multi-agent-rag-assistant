/**
 * app.js — Reactive Client for NexusCore Multi-Agent Assistant.
 * Controls real-time DAG node animations, chat execution, HITL approval modal,
 * Knowledge Base indexing, and benchmark evaluation suite.
 */

document.addEventListener("DOMContentLoaded", () => {
  const state = {
    sessionId: "exec-" + Math.random().toString(36).substring(2, 9),
    activeHitlSession: null,
    isProcessing: false
  };

  // DOM Elements
  const messagesStream = document.getElementById("messages-stream");
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const btnSend = document.getElementById("btn-send");
  const currentTierBadge = document.getElementById("current-tier-badge");
  const btnPendingHitl = document.getElementById("btn-pending-hitl");
  const hitlModal = document.getElementById("hitl-modal");
  const docModal = document.getElementById("doc-modal");

  // Telemetry Metric Elements
  const metricLatency = document.getElementById("metric-latency");
  const metricSavings = document.getElementById("metric-savings");
  const metricGrounding = document.getElementById("metric-grounding");
  const metricDefense = document.getElementById("metric-defense");

  // -------------------------------------------------------------
  // Tab Navigation
  // -------------------------------------------------------------
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");

      if (targetId === "kb-tab") loadDocuments();
      if (targetId === "telemetry-tab") loadTelemetry();
    });
  });

  // -------------------------------------------------------------
  // Preset Query Chips
  // -------------------------------------------------------------
  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      userInput.value = chip.getAttribute("data-query");
      chatForm.dispatchEvent(new Event("submit"));
    });
  });

  // -------------------------------------------------------------
  // DAG Node State Animation
  // -------------------------------------------------------------
  const dagNodes = [
    "node-input_guard",
    "node-pareto_router",
    "node-supervisor",
    "node-crag_specialist",
    "node-synthesizer",
    "node-auditor",
    "node-hitl"
  ];

  function resetDag() {
    dagNodes.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.className = "dag-node";
    });
  }

  function highlightDagNode(nodeId, status = "active") {
    const el = document.getElementById(nodeId);
    if (el) {
      el.className = `dag-node ${status}`;
    }
  }

  async function animateDagSequence(events, requiresHitl, isBlocked) {
    resetDag();
    if (isBlocked) {
      highlightDagNode("node-input_guard", "interrupted");
      return;
    }

    const mapping = {
      "input_guard": "node-input_guard",
      "pareto_router": "node-pareto_router",
      "supervisor": "node-supervisor",
      "crag_specialist": "node-crag_specialist",
      "synthesizer": "node-synthesizer",
      "auditor": "node-auditor",
      "hitl": "node-hitl"
    };

    for (let ev of events) {
      const elementId = mapping[ev.node];
      if (elementId) {
        highlightDagNode(elementId, "active");
        await new Promise(r => setTimeout(r, 80));
        highlightDagNode(elementId, "success");
      }
    }

    if (requiresHitl) {
      highlightDagNode("node-hitl", "interrupted");
    }
  }

  // -------------------------------------------------------------
  // Chat Workflow Execution
  // -------------------------------------------------------------
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query || state.isProcessing) return;

    state.isProcessing = true;
    userInput.value = "";
    btnSend.disabled = true;

    // Append User Message
    appendMessage("user", "EXECUTIVE CLIENT", query);

    // Initial DAG pulse
    resetDag();
    highlightDagNode("node-input_guard", "active");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, query: query })
      });

      const data = await res.json();
      
      // Update Model Tier Badge
      currentTierBadge.textContent = `${data.model_tier}: ${data.selected_model}`;
      
      // Animate DAG
      const isBlocked = data.status === "error";
      await animateDagSequence(data.step_events || [], data.requires_hitl, isBlocked);

      // Update Telemetry Bar
      if (data.step_events) {
        const totalLat = data.step_events.reduce((acc, cur) => acc + (cur.latency_ms || 0), 0);
        metricLatency.textContent = `${totalLat.toFixed(1)} ms`;
      }
      metricGrounding.textContent = `${(data.grounding_score * 100).toFixed(1)}%`;

      // Render Assistant Response
      if (data.requires_hitl) {
        appendHitlInterruptionMessage(data);
        checkPendingHitl();
      } else {
        appendAssistantMessage(data);
      }

    } catch (err) {
      appendMessage("assistant", "NEXUSCORE ERROR", `System fault during workflow execution: ${err.message}`);
    } finally {
      state.isProcessing = false;
      btnSend.disabled = false;
    }
  });

  function appendMessage(role, sender, text) {
    const card = document.createElement("div");
    card.className = `message-card ${role}`;
    card.innerHTML = `
      <div class="msg-header">
        <span class="msg-sender">${sender}</span>
        <span class="msg-meta">${new Date().toLocaleTimeString()}</span>
      </div>
      <div class="msg-body">${escapeHtml(text)}</div>
    `;
    messagesStream.appendChild(card);
    messagesStream.scrollTop = messagesStream.scrollHeight;
  }

  function appendAssistantMessage(data) {
    const card = document.createElement("div");
    card.className = "message-card assistant";

    // Format citations
    let formattedBody = escapeHtml(data.final_answer || data.draft_answer);
    if (data.citations && data.citations.length > 0) {
      data.citations.forEach(cit => {
        formattedBody = formattedBody.replaceAll(
          cit,
          `<span class="citation-pill" title="Verified source citation">${cit}</span>`
        );
      });
    }

    // Step Logs
    let stepsHtml = "";
    if (data.step_events && data.step_events.length > 0) {
      stepsHtml = data.step_events.map(ev => 
        `<div class="step-log-line">→ [${ev.node.toUpperCase()}] ${escapeHtml(ev.action)} (${ev.latency_ms}ms)</div>`
      ).join("");
    }

    card.innerHTML = `
      <div class="msg-header">
        <span class="msg-sender">NEXUSCORE (${data.selected_model || "Cognitive Agent"})</span>
        <span class="msg-meta">Grounding: ${(data.grounding_score*100).toFixed(0)}% | Risk: ${data.hallucination_risk}</span>
      </div>
      <div class="msg-body">${formattedBody}</div>
      <div class="trace-accordion">
        <button class="trace-toggle" onclick="toggleTrace(this)">
          <span>▶ View Cognitive Execution Trace (${data.step_events?.length || 0} nodes)</span>
        </button>
        <div class="trace-details">
          ${stepsHtml}
        </div>
      </div>
    `;
    messagesStream.appendChild(card);
    messagesStream.scrollTop = messagesStream.scrollHeight;
  }

  function appendHitlInterruptionMessage(data) {
    const card = document.createElement("div");
    card.className = "message-card assistant";
    card.innerHTML = `
      <div class="msg-header">
        <span class="msg-sender" style="color: var(--accent-amber);">⚠️ HUMAN-IN-THE-LOOP INTERRUPT</span>
        <span class="msg-meta">Paused for Authorization</span>
      </div>
      <div class="msg-body">
        <strong>Operation Paused:</strong> ${escapeHtml(data.hitl_reason)}<br><br>
        <em>Proposed Output:</em> "${escapeHtml(data.draft_answer)}"<br><br>
        <button class="btn-primary" style="padding: 6px 14px; font-size: 0.8rem;" onclick="openHitlReview('${data.session_id}', '${escapeHtml(data.hitl_reason)}', '${escapeHtml(data.draft_answer)}')">
          Review & Authorize in Executive Gateway
        </button>
      </div>
    `;
    messagesStream.appendChild(card);
    messagesStream.scrollTop = messagesStream.scrollHeight;
  }

  // -------------------------------------------------------------
  // Human-in-the-Loop Gateway
  // -------------------------------------------------------------
  window.openHitlReview = (sessionId, reason, draftAnswer) => {
    state.activeHitlSession = sessionId;
    document.getElementById("hitl-reason-text").textContent = reason;
    document.getElementById("hitl-query-preview").textContent = `Session: ${sessionId}`;
    document.getElementById("hitl-answer-editor").value = draftAnswer;
    hitlModal.classList.add("open");
  };

  document.getElementById("btn-close-hitl").addEventListener("click", () => hitlModal.classList.remove("open"));

  async function sendHitlDecision(action) {
    if (!state.activeHitlSession) return;
    const modifiedAnswer = document.getElementById("hitl-answer-editor").value;

    try {
      const res = await fetch("/api/hitl/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: state.activeHitlSession,
          action: action,
          modified_answer: modifiedAnswer
        })
      });

      const data = await res.json();
      hitlModal.classList.remove("open");
      appendMessage("assistant", `HITL GATEWAY [${action}]`, data.final_answer);
      checkPendingHitl();
    } catch (e) {
      alert("Error resolving HITL action: " + e.message);
    }
  }

  document.getElementById("btn-hitl-approve").addEventListener("click", () => sendHitlDecision("APPROVE"));
  document.getElementById("btn-hitl-edit").addEventListener("click", () => sendHitlDecision("EDIT"));
  document.getElementById("btn-hitl-reject").addEventListener("click", () => sendHitlDecision("REJECT"));

  async function checkPendingHitl() {
    try {
      const res = await fetch("/api/hitl/pending");
      const data = await res.json();
      const count = data.pending ? data.pending.length : 0;
      document.querySelector(".hitl-indicator").textContent = count;
    } catch (e) {}
  }

  // -------------------------------------------------------------
  // Knowledge Base Documents
  // -------------------------------------------------------------
  async function loadDocuments() {
    const grid = document.getElementById("docs-grid");
    grid.innerHTML = "<p>Loading knowledge base index...</p>";

    try {
      const res = await fetch("/api/documents");
      const data = await res.json();
      
      grid.innerHTML = "";
      data.documents.forEach(doc => {
        const card = document.createElement("div");
        card.className = "doc-card";
        card.innerHTML = `
          <div class="doc-title">📄 ${escapeHtml(doc.title)}</div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">${doc.char_count} Characters Indexed</div>
          <div class="doc-stats">
            <span>Chunks: <strong>${doc.chunk_count}</strong></span>
            <span>Category: <strong>${doc.metadata?.category || "General"}</strong></span>
          </div>
        `;
        grid.appendChild(card);
      });
    } catch (e) {
      grid.innerHTML = `<p style="color: var(--accent-red);">Failed to load documents: ${e.message}</p>`;
    }
  }

  document.getElementById("btn-open-add-doc").addEventListener("click", () => docModal.classList.add("open"));
  document.getElementById("btn-close-doc").addEventListener("click", () => docModal.classList.remove("open"));
  document.getElementById("btn-cancel-doc").addEventListener("click", () => docModal.classList.remove("open"));

  document.getElementById("btn-submit-doc").addEventListener("click", async () => {
    const title = document.getElementById("doc-title-input").value.trim();
    const content = document.getElementById("doc-content-input").value.trim();
    if (!title || !content) {
      alert("Please provide both title and content.");
      return;
    }

    try {
      await fetch("/api/documents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, content, category: "Custom" })
      });
      docModal.classList.remove("open");
      loadDocuments();
    } catch (e) {
      alert("Failed to index document: " + e.message);
    }
  });

  // -------------------------------------------------------------
  // Evaluation Benchmark Suite
  // -------------------------------------------------------------
  const btnRunEvals = document.getElementById("btn-run-evals-live");
  btnRunEvals.addEventListener("click", async () => {
    btnRunEvals.disabled = true;
    btnRunEvals.innerHTML = "Executing Benchmarks...";

    try {
      const res = await fetch("/api/evals/run");
      const summary = await res.json();

      document.getElementById("eval-stat-pass").textContent = `${summary.pass_rate_percent}%`;
      document.getElementById("eval-stat-grounding").textContent = `${summary.avg_grounding_score_percent}%`;
      document.getElementById("eval-stat-defense").textContent = `${summary.injection_defense_rate_percent}%`;
      document.getElementById("eval-stat-latency").textContent = `${summary.avg_latency_ms} ms`;

      const tbody = document.getElementById("eval-table-body");
      tbody.innerHTML = "";
      summary.cases.forEach(c => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><code>${c.id}</code></td>
          <td><strong>${c.category}</strong></td>
          <td><span style="color: ${c.passed ? 'var(--accent-green)' : 'var(--accent-red)'}; font-weight: 700;">${c.passed ? '✅ PASS' : '❌ FAIL'}</span></td>
          <td>${c.latency_ms} ms</td>
          <td>${(c.grounding_score * 100).toFixed(0)}%</td>
          <td style="color: var(--text-muted);">${escapeHtml(c.notes)}</td>
        `;
        tbody.appendChild(row);
      });
    } catch (e) {
      alert("Error running evals: " + e.message);
    } finally {
      btnRunEvals.disabled = false;
      btnRunEvals.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Full Benchmark Suite`;
    }
  });

  // -------------------------------------------------------------
  // Telemetry Observability Tab
  // -------------------------------------------------------------
  async function loadTelemetry() {
    const list = document.getElementById("traces-list");
    list.innerHTML = "<p>Loading recent execution traces...</p>";

    try {
      const res = await fetch("/api/telemetry");
      const data = await res.json();
      
      list.innerHTML = "";
      if (!data.recent_traces || data.recent_traces.length === 0) {
        list.innerHTML = "<p style='color: var(--text-faint);'>No execution traces recorded yet. Execute queries to see telemetry.</p>";
        return;
      }

      data.recent_traces.forEach(t => {
        const card = document.createElement("div");
        card.className = "trace-card";
        card.innerHTML = `
          <div class="trace-card-header">
            <span>TRACE ID: ${t.trace_id.substring(0, 8)}...</span>
            <span>${new Date(t.start_time * 1000).toLocaleTimeString()} | Total: ${t.total_latency_ms} ms</span>
          </div>
          <div style="font-size: 0.9rem; margin-bottom: 8px;"><strong>Query:</strong> "${escapeHtml(t.user_query)}"</div>
          <div style="font-size: 0.8rem; color: var(--accent-cyan);">Pareto Cost Saved: $${t.total_cost_saved_usd.toFixed(6)} | Tokens: ${t.total_tokens}</div>
        `;
        list.appendChild(card);
      });
    } catch (e) {
      list.innerHTML = `<p style="color: var(--accent-red);">Failed to load telemetry: ${e.message}</p>`;
    }
  }

  document.getElementById("btn-refresh-telemetry").addEventListener("click", loadTelemetry);

  // Helper Utilities
  window.toggleTrace = (btn) => {
    const details = btn.nextElementSibling;
    details.classList.toggle("open");
  };

  function escapeHtml(text) {
    if (!text) return "";
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Pre-fetch initial evals table
  btnRunEvals.click();
});
