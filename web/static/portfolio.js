"use strict";

(() => {
  const API = "/api/v2/offline-portfolio";
  const $ = (id) => document.getElementById(id);
  const state = {
    bundles: [], bundle: null, snapshot: null, analysis: null, selected: new Set(), industries: {},
    saved: null, whitelists: [], plan: null, previousPlan: null, previousInput: null, epoch: 0, edit: 0, listEpoch: 0,
    loading: false, saving: false, generating: false, importing: false,
  };
  const ACTIONS = { TARGET_ONLY: "仅目标", BUY: "计划买入", SELL: "计划卖出", HOLD: "保持" };
  const EXECUTION = {
    UNKNOWN: "执行信息未知", BLOCKED_BUY: "买入受阻", BLOCKED_SELL: "卖出受阻",
    READY: "模拟可执行", NO_CHANGE: "无需调仓", PARTIAL_CASH_LIMIT: "现金限制，部分可执行",
    PARTIAL_RISK_LIMIT: "受保留持仓风险限额限制",
    NO_TRADE: "费用或效用不支持交易", BLOCKED_LIQUIDITY: "流动性不足",
    PARTIAL_RISK_OR_CAPACITY_LIMIT: "风险或容量限制，部分执行",
    PARTIAL_LOT_OR_SELLABLE_LIMIT: "交易单位或可卖数量限制",
    BUY_PAUSED_ROUNDING_RISK: "取整后风险超限，暂停买入",
    BUY_PAUSED_INFEASIBLE_EXECUTION: "执行约束冲突，暂停买入",
    LIQUIDITY_INFORMATION_MISSING: "缺少流动性资料，暂停新增",
  };
  const REASONS = {
    SYNTHETIC_DATA: "合成数据，仅供功能演示。", SYNTHETIC_ONLY: "合成数据，仅供功能演示。",
    SYNTHETIC_DATA_ONLY: "合成数据，仅供功能演示。",
    SYNTHETIC_DATA_NOT_MARKET_EVIDENCE: "合成数据不代表真实市场结果或模型有效性证据。",
    OFFLINE_DEMO_NOT_A_VALIDATED_STRATEGY: "离线演示尚未通过策略有效性验证。",
    HISTORICAL_REPLAY_NOT_LIVE_PREDICTION: "仅回放已保存的历史评分，不提供实时预测。",
    DEFAULT_BUNDLE_REJECTED: "一个默认数据包未通过导入检查；请检查文件后重新导入。",
    INCOMPLETE_UNIVERSE: "当前数据包的股票数未达到完整 300 只。",
    MODEL_NOT_VALIDATED: "模型尚未通过正式有效性验证。",
    MISSING_INDUSTRY: "未提供行业的股票共用未知行业的 30% 上限。",
    UNKNOWN_INDUSTRY: "未知行业统一受 30% 上限约束。",
    MISSING_HOLDINGS: "未提供当前持仓，仅展示目标比例。",
    NO_VALID_SCORE: "没有有效评分。", INVALID_SCORE: "评分无效。",
    NO_TRADABLE_WEIGHTS: "未生成正式交易权重。",
    OFFLINE_ONLY: "仅用于离线模拟。", HOLDOUT_NOT_VALIDATED: "尚未完成正式 Holdout 验证。",
    SYNTHETIC_UNIVERSE_NOT_300: "合成包未包含完整 300 只股票。",
    INVALID_MEMBER_SCORES_PRESERVED: "缺分股票保留展示；如选入白名单，整份建议将报告缺分冲突。",
    INVALID_OR_UNAVAILABLE_SCORE: "评分缺失、无效或当时尚不可用。",
    OFFLINE_SIMULATION_ONLY: "仅用于离线配置模拟。",
    SCORES_ARE_NOT_EXPECTED_RETURNS: "评分不等于预期收益或收益概率。",
    UNKNOWN_INDUSTRY_SHARED_30_PERCENT_CAP: "未填行业的股票共用未知行业的 30% 上限。",
    NO_VALID_SELECTED_SCORE: "所选股票没有有效评分，目标为全部现金。",
    MISSING_HOLDINGS_TARGET_ONLY: "未提供持仓，仅生成目标权重。",
    CASH_RESERVED_FOR_ESTIMATED_COST: "已在目标现金中预留预计交易费用。",
    UNUSED_BUDGET_REMAINS_CASH: "受约束而未分配的预算保留为现金。",
    BUY_EXECUTION_BLOCKED: "部分股票买入受阻，无法按目标买入。",
    BUYS_LIMITED_BY_AVAILABLE_CASH: "可用现金不足，模拟买入已相应缩减。",
    BUYS_LIMITED_BY_RETAINED_HOLDING_RISK: "无法卖出的旧仓占用风险限额，模拟新买已相应缩减。",
    EXECUTION_BLOCKED_OLD_HOLDINGS_RETAINED: "部分旧持仓卖出受阻，模拟中继续保留。",
    PROJECTED_HOLDINGS_EXCEED_TARGET_LIMITS_DUE_TO_BLOCKS: "因旧仓无法卖出，模拟执行后的持仓仍可能超过目标风险限制。",
    EXECUTION_INFORMATION_MISSING: "缺少执行信息，无法推算完成调仓后的持仓和现金。",
    NOT_IN_WHITELIST_EXIT_TARGET: "原持仓不在当前白名单内，目标权重为零。",
    INVALID_SCORE_NO_POSITIVE_TARGET: "评分无效，目标权重为零。",
    SELECTED_TARGETS_STRICTLY_POSITIVE: "全部所选股票的推荐目标均大于零。",
    RANKING_SCORE_DIFFERS_FROM_PORTFOLIO_UTILITY: "排名使用股票分数，配置比较组合收益、风险与成本。",
    NO_ACCOUNT_MINIMUM_COMMISSION_LOTS_AND_CAPACITY_UNCHECKED: "未提供账户净值：未验证最低佣金、股数及交易金额容量。",
    SMALL_REBALANCES_FROZEN: "合法旧仓的小幅调整已保留原权重。",
    DEADBAND_SKIPPED_FOR_HARD_CONSTRAINTS: "保持小额调仓会违反硬约束，已按约束重新求解。",
    SHARE_QUANTITIES_UNAVAILABLE_WEIGHT_SIMULATION_ONLY: "缺少完整股数或交易单位，只能模拟比例。",
    SELECTED_POSITIVE_HOLDINGS_NOT_EXECUTABLE: "部分所选股票的正目标无法完成建仓；查看逐股执行状态。",
    REQUEST_NOT_FULLY_EXECUTED: "完整白名单建仓要求尚未满足。",
    NO_TRADE_COST_OR_UTILITY: "本次交易的保守收益不足以补偿成本和风险，模拟保持原仓。",
    RETAINED_HOLDINGS_EXCEED_RISK_LIMITS: "保留旧仓仍超过风险限制；暂停新增并披露超限。",
    EXECUTION_CONSTRAINT_INFEASIBLE: "目标可行，但当前执行条件下无法完整调仓。",
    ALLOCATION_CONSTRAINT_INFEASIBLE: "正权重、风险、费用或预算约束冲突。",
    ROUNDING_RECHECK_REQUIRED: "股数取整后已重新检查风险和资金。",
    AT_USER_MINIMUM_WEIGHT: "目标处于你设置的最低权重。", AT_STOCK_CAP: "目标达到单股5%上限。",
  };

  function node(tag, content, className) {
    const el = document.createElement(tag);
    if (content !== undefined && content !== null) el.textContent = String(content);
    if (className) el.className = className;
    return el;
  }

  function notify(message, type = "info") {
    $("message").textContent = message;
    $("message").className = "message " + type;
    $("message").hidden = false;
    if (type === "error") $("message").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function validScore(row) {
    return row.signal_valid === true && row.score !== null && row.score !== undefined
      && Number.isFinite(Number(row.score));
  }

  function percent(value, signed = false) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const num = Number(value) * 100;
    return (signed && num > 0 ? "+" : "") + num.toLocaleString("zh-CN", { maximumFractionDigits: Math.abs(num) < .01 ? 5 : 3 }) + "%";
  }

  function synthetic(role) {
    return String(role || "").toLowerCase().includes("synthetic");
  }

  function roleLabel(role) {
    return synthetic(role) ? "合成示例 · 非真实行情" : "离线历史数据 · 未验证";
  }

  async function request(path, options = {}) {
    let response;
    try {
      response = await fetch(API + path, {
        ...options, headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}) },
        cache: "no-store",
      });
    } catch (_error) {
      throw new Error("无法连接本地服务，请确认网页服务仍在运行后重试。");
    }
    let data;
    try { data = await response.json(); } catch (_error) {
      throw new Error("服务未返回可读取的数据，请确认离线配置接口已启动。");
    }
    if (!response.ok) {
      let detail = data.detail || data.message || "请求未成功";
      if (Array.isArray(detail)) {
        const oldBackend = detail.some((item) => item.type === "extra_forbidden" && item.loc?.includes("range_utility_tolerance_bps"));
        detail = oldBackend ? "当前后台进程仍是旧版本。请停止旧服务，重新运行 run_portfolio.py 后刷新页面，以加载新版区间算法。"
          : detail.map((item) => item.msg || "输入格式不正确").join("；");
      }
      if (detail && typeof detail === "object" && detail.message) {
        detail = detail.message + (detail.details?.codes ? "：" + detail.details.codes.join("、") : "")
          + " [" + detail.code + "]";
      }
      if (typeof detail !== "string") detail = "输入格式不正确";
      if (response.status === 409) detail = "保存的版本已变化，请重新读取白名单后再保存。";
      throw new Error(detail);
    }
    return data;
  }

  function warnings(target, items) {
    const list = $(target);
    list.replaceChildren();
    const texts = [...new Set((items || []).filter(Boolean).map((item) => {
      if (typeof item === "string") return REASONS[item] || item;
      return String(item.message || item.reason || item.code || "数据包包含额外说明。");
    }))];
    for (const text of texts) list.append(node("li", text));
    list.hidden = texts.length === 0;
  }

  function addOption(select, value, label) {
    const option = node("option", label);
    option.value = value;
    select.append(option);
  }

  function setOptions(select, rows, emptyText) {
    select.replaceChildren();
    if (!rows.length) addOption(select, "", emptyText || "暂无选项");
    for (const row of rows) addOption(select, row.value, row.label);
  }

  function sameSelection(saved = state.saved) {
    if (!saved || !state.snapshot || saved.snapshot_id !== state.snapshot.snapshot_id) return false;
    const codes = saved.codes || [];
    return saved.name === $("whitelistName").value.trim() && codes.length === state.selected.size
      && codes.every((code) => state.selected.has(code));
  }

  function invalidatePlan(reason) {
    state.edit += 1;
    state.plan = null;
    $("planResult").hidden = true;
    $("planEmpty").hidden = false;
    $("planEmpty").querySelector("p").textContent = reason || "保存白名单并生成方案后，查看目标权重、现金和调仓差额。";
    $("downloadPlan").disabled = true;
    updateControls();
  }

  function updateControls() {
    const ready = !!state.snapshot && !state.loading;
    const saved = ready && sameSelection();
    $("saveWhitelist").disabled = !ready || !state.selected.size || state.saving;
    $("saveAsNew").disabled = !ready || !state.selected.size || state.saving;
    $("saveAsNew").hidden = !state.saved;
    const minimumReady = $("minimumWeight").value !== "" && Number($("minimumWeight").value) > 0;
    $("generatePlan").disabled = !saved || !state.analysis?.available || !minimumReady || state.generating || state.saving;
    $("selectVisible").disabled = !ready;
    $("clearSelection").disabled = !ready || !state.selected.size;
    $("reloadScores").disabled = state.loading || !state.bundle || !$("dateSelect").value;
    $("restoreWhitelist").disabled = !$("savedWhitelistSelect").value || state.loading;
    $("importBundle").disabled = state.importing;
    $("selectedCount").textContent = state.selected.size;
    $("saveWhitelist").textContent = state.saving ? "正在保存…" : state.saved ? "保存新版本" : "保存白名单";
    $("generatePlan").firstChild.textContent = state.generating ? "正在计算… " : "生成离线配置方案 ";
    if (saved) {
      $("saveState").textContent = "已保存 · " + state.saved.name + " · v" + state.saved.version + " · " + state.selected.size + " 只股票";
      $("planPrerequisite").textContent = "方案绑定已保存的 v" + state.saved.version + " 与当前历史评分快照。";
    } else if (state.snapshot) {
      $("saveState").textContent = "当前选择或名称尚未保存。保存后可生成配置方案。";
      $("planPrerequisite").textContent = "先保存当前白名单。";
    } else {
      $("saveState").textContent = "勾选股票后保存，方案将绑定这个名单版本和历史评分快照。";
      $("planPrerequisite").textContent = "先载入历史评分并保存白名单。";
    }
    if (saved && !state.analysis?.available) $("planPrerequisite").textContent = "当前快照缺少可用配置证据，请先导入。";
    else if (saved && !minimumReady) $("planPrerequisite").textContent = "请填写每只所选股票的最低目标权重。";
    const maxStock = Math.min(1, state.selected.size * 0.05);
    $("capacityNote").textContent = "已选 " + state.selected.size + " 只，单股上限合计最多 " + percent(maxStock)
      + "。行业与总预算可能进一步降低仓位；未填行业的股票共用「未知行业」30% 上限。";
    const minimum = Number($("minimumWeight").value);
    $("boundsHint").textContent = minimum >= 5
      ? "权重已锁定：最低目标5%与单股上限5%相同。每只只能配5%，区间也固定；降低最低目标才有优化空间。"
      : "预算是仓位上限，并不要求用满。上限未触及时放宽参数，目标可能保持不变；结果会解释实际约束。";
    $("boundsHint").className = minimum >= 5 ? "helper blocked" : "helper";
  }

  function beginContext() {
    state.epoch += 1;
    state.loading = true;
    $("message").hidden = true;
    $("scoreSource").textContent = "正在读取当前数据包的评分来源…";
    $("tableCaption").textContent = "正在载入全量股票与评分，载入完成后按有效评分降序展示。";
    state.snapshot = null;
    state.analysis = null;
    $("analysisStatus").textContent = "等待当前快照的配置证据…";
    $("analysisDetails").textContent = "";
    $("validationSummary").textContent = "读取当前快照的复核记录…";
    $("validationDetails").textContent = "";
    state.selected = new Set();
    state.industries = {};
    state.saved = null;
    invalidatePlan("历史评分快照已切换，请重新选择或读取白名单后生成方案。");
    $("memberCount").textContent = "—";
    $("validCount").textContent = "—";
    $("universeBadge").textContent = "正在加载";
    $("snapshotMeta").textContent = "正在读取已保存的历史评分…";
    $("stockRows").replaceChildren();
    const tr = node("tr");
    const td = node("td", "正在加载历史评分…", "empty-cell");
    td.colSpan = 6;
    tr.append(td);
    $("stockRows").append(tr);
    $("visibleCount").textContent = "0 只股票";
    warnings("sourceWarnings", []);
    updateControls();
    return state.epoch;
  }

  function contextFailure(error, epoch) {
    if (epoch !== state.epoch) return;
    state.loading = false;
    $("universeBadge").textContent = "加载失败";
    $("snapshotMeta").textContent = "历史评分未载入，请重新选择数据包或重试。";
    const tr = node("tr");
    const td = node("td", "未能读取评分。可点击重新载入重试。", "empty-cell");
    td.colSpan = 6;
    tr.append(td);
    $("stockRows").replaceChildren(tr);
    notify(error.message, "error");
    updateControls();
  }

  function fillPeriods(bundle, preferredDate, preferredFold) {
    const periods = bundle.periods || [];
    const dates = [...new Set(periods.map((p) => p.date))].sort().reverse();
    setOptions($("dateSelect"), dates.map((date) => ({ value: date, label: date + " · 历史" })), "暂无保存日期");
    if (dates.includes(preferredDate)) $("dateSelect").value = preferredDate;
    $("dateSelect").disabled = !dates.length;
    fillFolds(preferredFold);
  }

  function fillFolds(preferredFold) {
    const date = $("dateSelect").value;
    const periods = (state.bundle?.periods || []).filter((p) => p.date === date);
    setOptions($("foldSelect"), periods.map((p) => ({ value: p.fold_id, label: p.fold_id })), "暂无验证折");
    if (periods.some((p) => p.fold_id === preferredFold)) $("foldSelect").value = preferredFold;
    $("foldSelect").disabled = !periods.length;
  }

  function sortedRows() {
    return [...(state.snapshot?.rows || [])].sort((a, b) => {
      const validA = validScore(a), validB = validScore(b);
      if (validA !== validB) return validA ? -1 : 1;
      if (validA && Number(a.score) !== Number(b.score)) return Number(b.score) - Number(a.score);
      return String(a.code).localeCompare(String(b.code));
    });
  }

  function visibleRows() {
    const query = $("stockSearch").value.trim().toUpperCase();
    return sortedRows().filter((row) => String(row.code).toUpperCase().includes(query));
  }

  function renderStocks() {
    const rows = visibleRows();
    const fragment = document.createDocumentFragment();
    for (const row of rows) {
      const tr = node("tr", null, state.selected.has(row.code) ? "selected" : "");
      const tdCheck = node("td");
      const checkbox = node("input");
      checkbox.type = "checkbox";
      checkbox.checked = state.selected.has(row.code);
      checkbox.setAttribute("aria-label", "选择股票 " + row.code);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) state.selected.add(row.code); else state.selected.delete(row.code);
        tr.classList.toggle("selected", checkbox.checked);
        invalidatePlan("股票选择已更改，请保存白名单后重新生成方案。");
      });
      tdCheck.append(checkbox);
      tr.append(tdCheck, node("td", validScore(row) ? row.rank ?? "—" : "—"), node("td", row.code));
      tr.append(node("td", validScore(row) ? Number(row.score).toFixed(5) : "—", "numeric score-cell"));
      const status = node("td");
      const statusText = node("span", validScore(row) ? "评分有效" : "缺分 / 不可配置", "row-status" + (validScore(row) ? "" : " invalid"));
      statusText.title = (row.reason_codes || []).map((code) => REASONS[code] || code).join("；");
      status.append(statusText);
      tr.append(status);
      const industry = node("td");
      const input = node("input", null, "industry-input");
      input.type = "text";
      input.maxLength = 64;
      input.placeholder = "未知行业";
      input.value = state.industries[row.code] || "";
      input.readOnly = Boolean(state.analysis?.market?.[row.code]?.industry);
      input.setAttribute("aria-label", row.code + " 的行业");
      input.addEventListener("input", () => {
        if (input.value.trim()) state.industries[row.code] = input.value.trim();
        else delete state.industries[row.code];
        invalidatePlan("行业假设已更改，请重新生成方案。");
      });
      industry.append(input);
      tr.append(industry);
      fragment.append(tr);
    }
    if (!rows.length) {
      const tr = node("tr");
      const td = node("td", state.snapshot ? "没有匹配的股票代码。清空搜索可查看全量股票。" : "请先选择历史评分。", "empty-cell");
      td.colSpan = 6;
      tr.append(td);
      fragment.append(tr);
    }
    $("stockRows").replaceChildren(fragment);
    $("visibleCount").textContent = "显示 " + rows.length + " / 全量 " + (state.snapshot?.rows?.length || 0) + " 只";
    updateControls();
  }

  function applySnapshot(snapshot, selected = [], saved = null) {
    if (!snapshot.snapshot_id || !Array.isArray(snapshot.rows)) throw new Error("评分快照字段不完整，请检查数据包。");
    const codes = new Set(snapshot.rows.map((row) => row.code));
    if (selected.some((code) => !codes.has(code))) throw new Error("白名单含有原评分快照中不存在的股票，无法恢复。");
    state.snapshot = snapshot;
    state.loading = false;
    state.selected = new Set(selected);
    state.saved = saved;
    if (saved) $("whitelistName").value = saved.name;
    $("stockSearch").value = "";
    const members = snapshot.member_count ?? snapshot.rows.length;
    const expected = snapshot.expected_member_count || 300;
    const valid = snapshot.rows.filter(validScore).length;
    $("memberCount").textContent = members;
    $("validCount").textContent = valid;
    $("dataRole").textContent = roleLabel(snapshot.data_role);
    $("dataRole").className = "badge" + (synthetic(snapshot.data_role) ? " warning" : "");
    $("universeBadge").textContent = snapshot.is_complete_universe ? "全量 " + members + " 只" : "样本 " + members + " / " + expected;
    $("universeBadge").className = "badge" + (snapshot.is_complete_universe ? " good" : " warning");
    $("snapshotMeta").textContent = "历史评分：" + snapshot.date + " · " + snapshot.fold_id
      + " · " + (synthetic(snapshot.data_role) ? "合成示例，不代表真实当期成分" : "来自已保存的离线数据包")
      + " · 评分可用时间 " + (snapshot.available_at || "未提供") + " · 快照 " + snapshot.snapshot_id;
    const output = state.bundle?.signal_output;
    const scoreMethod = output === "model_prediction" ? "选中公式或特征经拟合后输出的原始模型评分"
      : output === "equal_weight" ? "选中公式的横截面排名等权合成评分" : "数据包保存的研究评分（未声明合成方法）";
    $("scoreSource").textContent = "评分来源：" + scoreMethod + "。仅回放已保存日期，不代表已有可复用的新日期预测模型。";
    $("scoreColumn").textContent = output === "model_prediction" ? "模型原始评分 ↓" : "公式组合评分 ↓";
    $("tableCaption").textContent = "按 " + snapshot.date + " 的有效评分降序展示全部 " + snapshot.rows.length
      + " 只（不是 Top-N）；缺分股票保留在末尾。";
    // These facts are already displayed by the page notice, data-role badge and source text.
    const explained = new Set(["OFFLINE_DEMO_NOT_A_VALIDATED_STRATEGY", "HISTORICAL_REPLAY_NOT_LIVE_PREDICTION",
      "SYNTHETIC_DATA", "SYNTHETIC_ONLY", "SYNTHETIC_DATA_ONLY", "SYNTHETIC_DATA_NOT_MARKET_EVIDENCE",
      "SYNTHETIC_UNIVERSE_NOT_300", "INCOMPLETE_UNIVERSE"]);
    const notes = [...(state.bundle?.warnings || []), ...(snapshot.warnings || [])].filter((item) => !explained.has(item));
    if (!snapshot.is_complete_universe) notes.push("实际仅有 " + members + " 只股票，未达到 300 只；不会补造缺失评分。");
    warnings("sourceWarnings", notes);
    renderStocks();
    void loadAnalysis(snapshot.snapshot_id, state.epoch);
  }

  async function loadAnalysis(snapshotId, epoch) {
    try {
      const result = await request("/analysis?" + new URLSearchParams({ snapshot_id: snapshotId }));
      if (epoch !== state.epoch || snapshotId !== state.snapshot?.snapshot_id) return;
      state.analysis = result;
      $("analysisStatus").textContent = result.available
        ? "配置证据已就绪 · " + result.horizon + " 日周期 · " + result.calibrator.date_count + " 个成熟校准日 · "
          + result.risk_period_count + " 个非重叠风险周期 · " + roleLabel(result.data_role)
        : result.message;
      $("analysisDetails").textContent = result.available ? JSON.stringify({ analysis_id: result.analysis_id,
        rule_version: result.rule_version, calibrator: result.calibrator, risk_method: result.risk_method,
        risk_end_at: result.risk_end_at, missing_risk_codes: result.missing_risk_codes }, null, 2) : result.message;
      for (const [code, info] of Object.entries(result.market || {})) if (info.industry) state.industries[code] = info.industry;
      renderStocks();
      const validation = await request("/validation?" + new URLSearchParams({ snapshot_id: snapshotId }));
      if (epoch !== state.epoch || snapshotId !== state.snapshot?.snapshot_id) return;
      const boundReports = validation.reports.filter((report) => report.analysis_ids.includes(result.analysis_id));
      const latest = boundReports[0];
      $("validationSummary").textContent = latest ? "当前证据已记录 " + boundReports.length + " 次回放；最近 "
        + latest.summaries.length + " 组方法／成本／相位结果。" + (latest.comparison_passed ? "研究比较通过，仍未发布生产。" : "研究证据不足或未达门槛，尚未通过策略验证。")
        : "当前快照尚无历史回放报告；生成单份方案仅检查约束与执行条件。";
      $("validationDetails").textContent = latest ? JSON.stringify(latest, null, 2) : "";
    } catch (error) {
      if (epoch === state.epoch) { notify(error.message, "error"); updateControls(); }
    }
  }

  async function fetchScores(epoch) {
    const bundleId = state.bundle?.bundle_id;
    const date = $("dateSelect").value;
    const fold = $("foldSelect").value;
    if (!bundleId || !date || !fold) throw new Error("数据包没有可用的历史日期或验证折。");
    const params = new URLSearchParams({ bundle_id: bundleId, date, fold_id: fold });
    const snapshot = await request("/scores?" + params);
    if (epoch !== state.epoch) return;
    applySnapshot(snapshot);
  }

  async function loadBundle(bundleId) {
    if (!bundleId) return;
    const epoch = beginContext();
    state.bundle = null;
    $("dateSelect").disabled = true;
    $("foldSelect").disabled = true;
    try {
      const bundle = await request("/bundles/" + encodeURIComponent(bundleId));
      if (epoch !== state.epoch) return;
      state.bundle = bundle;
      $("dataRole").textContent = roleLabel(bundle.data_role);
      fillPeriods(bundle);
      await fetchScores(epoch);
    } catch (error) { contextFailure(error, epoch); }
  }

  async function loadScores() {
    const epoch = beginContext();
    try { await fetchScores(epoch); } catch (error) { contextFailure(error, epoch); }
  }

  async function listBundles(preferredId) {
    const payload = await request("/bundles");
    state.bundles = payload.bundles || [];
    setOptions($("bundleSelect"), state.bundles.map((bundle) => ({ value: bundle.bundle_id,
      label: (bundle.label || bundle.bundle_id) + (synthetic(bundle.data_role) ? " · 合成示例" : " · 离线研究") })), "没有可用数据包，请导入");
    $("bundleSelect").disabled = !state.bundles.length;
    if (state.bundles.some((item) => item.bundle_id === preferredId)) $("bundleSelect").value = preferredId;
    return $("bundleSelect").value;
  }

  async function listWhitelists() {
    const epoch = ++state.listEpoch;
    const payload = await request("/whitelists");
    if (epoch !== state.listEpoch) return;
    state.whitelists = payload.whitelists || [];
    const previous = $("savedWhitelistSelect").value;
    $("savedWhitelistSelect").replaceChildren();
    addOption($("savedWhitelistSelect"), "", state.whitelists.length ? "选择一个已保存名单" : "尚无保存的白名单");
    for (const item of state.whitelists) {
      addOption($("savedWhitelistSelect"), item.whitelist_id, item.name + " · v" + item.version + " · " + (item.codes?.length ?? "—") + " 只");
    }
    if (state.whitelists.some((item) => item.whitelist_id === previous)) $("savedWhitelistSelect").value = previous;
    updateControls();
  }

  async function saveWhitelist(asNew = false) {
    if (!state.snapshot || state.saving || state.loading) return;
    const name = $("whitelistName").value.trim();
    if (!name) { notify("请先填写白名单名称。", "error"); $("whitelistName").focus(); return; }
    if (!state.selected.size) { notify("请至少选择一只股票。", "error"); return; }
    const epoch = state.epoch;
    const payload = { snapshot_id: state.snapshot.snapshot_id, name, codes: [...state.selected].sort() };
    if (state.saved && !asNew) {
      payload.whitelist_id = state.saved.whitelist_id;
      payload.expected_version = state.saved.version;
    }
    state.saving = true;
    updateControls();
    try {
      const saved = await request("/whitelists", { method: "POST", body: JSON.stringify(payload) });
      if (epoch === state.epoch) {
        state.saved = saved;
        invalidatePlan("白名单已保存为 v" + saved.version + "，可以生成离线配置方案。");
        notify(sameSelection() ? "白名单已保存 · v" + saved.version + "。" : "刚才提交的名单已保存；当前选择又有变化，请再次保存。", "success");
      }
      await listWhitelists();
      if (epoch === state.epoch) $("savedWhitelistSelect").value = saved.whitelist_id;
    } catch (error) { if (epoch === state.epoch) notify(error.message, "error"); }
    finally { state.saving = false; updateControls(); }
  }

  async function restoreWhitelist() {
    const id = $("savedWhitelistSelect").value;
    if (!id) return;
    const epoch = beginContext();
    try {
      const saved = await request("/whitelists/" + encodeURIComponent(id));
      if (epoch !== state.epoch) return;
      const snapshot = await request("/snapshots/" + encodeURIComponent(saved.snapshot_id));
      if (epoch !== state.epoch) return;
      const bundle = await request("/bundles/" + encodeURIComponent(snapshot.bundle_id));
      if (epoch !== state.epoch) return;
      state.bundle = bundle;
      if (!state.bundles.some((b) => b.bundle_id === snapshot.bundle_id)) {
        addOption($("bundleSelect"), snapshot.bundle_id, bundle.label || snapshot.bundle_id);
      }
      $("bundleSelect").value = snapshot.bundle_id;
      fillPeriods(bundle, snapshot.date, snapshot.fold_id);
      // A recovered immutable snapshot remains authoritative even if the bundle changes later.
      if ($("dateSelect").value !== snapshot.date) {
        addOption($("dateSelect"), snapshot.date, snapshot.date + " · 原快照");
        $("dateSelect").value = snapshot.date;
      }
      if ($("foldSelect").value !== snapshot.fold_id) {
        addOption($("foldSelect"), snapshot.fold_id, snapshot.fold_id + " · 原快照");
        $("foldSelect").value = snapshot.fold_id;
      }
      applySnapshot(snapshot, saved.codes, saved);
      notify("已恢复「" + saved.name + "」v" + saved.version + " 与 " + snapshot.date + " 的原评分快照。行业、持仓和成本请按本次模拟填写。", "success");
    } catch (error) { contextFailure(error, epoch); }
  }

  function numberInput(id, label, min, max, optional = false) {
    const raw = $(id).value.trim();
    if (!raw && optional) return undefined;
    const value = Number(raw);
    if (!raw || !Number.isFinite(value) || value < min || value > max) throw new Error(label + "须在 " + min + " 至 " + max + " 之间。");
    return value;
  }

  function parseBoolean(text, line) {
    if (!text) return undefined;
    const value = text.toLowerCase();
    if (["true", "1", "是", "可", "可交易"].includes(value)) return true;
    if (["false", "0", "否", "不可", "阻塞"].includes(value)) return false;
    throw new Error("持仓第 " + line + " 行的可买/可卖须填写 true、false 或留空。");
  }

  function parseHoldings() {
    const mode = $("holdingsMode").value;
    if (mode === "unknown") return null;
    if (mode === "cash") return [];
    const lines = $("holdingsText").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (!lines.length) throw new Error("请填写当前持仓，或选择「当前空仓」。");
    const seen = new Set();
    const holdings = lines.map((line, index) => {
      const fields = line.split(/[,，]/).map((field) => field.trim());
      if (fields.length < 2 || fields.length > 6) throw new Error("持仓第 " + (index + 1) + " 行格式不正确，应为代码,权重%,行业,可买,可卖,股数。");
      const [rawCode, rawWeight, industry = "", buy = "", sell = "", quantity = ""] = fields;
      const code = rawCode.toUpperCase();
      const weight = Number(rawWeight.replace(/%$/, ""));
      if (!code || seen.has(code)) throw new Error("持仓代码为空或重复：" + code);
      if (!rawWeight || !Number.isFinite(weight) || weight < 0 || weight > 100) throw new Error("持仓第 " + (index + 1) + " 行的权重须为 0 至 100 的百分数。");
      seen.add(code);
      const row = { code, current_weight: weight / 100 };
      if (industry) row.industry = industry;
      const canBuy = parseBoolean(buy, index + 1), canSell = parseBoolean(sell, index + 1);
      if (canBuy !== undefined) row.can_buy = canBuy;
      if (canSell !== undefined) row.can_sell = canSell;
      if (quantity !== "") {
        if (!Number.isSafeInteger(Number(quantity)) || Number(quantity) < 0) throw new Error("持股数量须为非负整数。");
        row.quantity = Number(quantity);
      }
      return row;
    });
    if (holdings.reduce((sum, item) => sum + item.current_weight, 0) > 1 + 1e-10) throw new Error("当前股票持仓比例合计不能超过 100%。");
    return holdings;
  }

  function planPayload() {
    if (!sameSelection()) throw new Error("请先保存当前白名单。");
    const explicitCosts = $("explicitCosts").checked;
    const payload = {
      whitelist_id: state.saved.whitelist_id, whitelist_version: state.saved.version,
      market_budget: numberInput("marketBudget", "股票总预算", 0, 100) / 100,
      holdings: parseHoldings(), industries: { ...state.industries },
      buy_cost: explicitCosts ? .00076 : numberInput("buyCost", "买入成本", 0, 10) / 100,
      sell_cost: explicitCosts ? .00126 : numberInput("sellCost", "卖出成本", 0, 10) / 100,
      assume_tradable: $("assumeTradable").checked,
      minimum_weight: numberInput("minimumWeight", "每只最低目标", .0001, 5) / 100,
      volatility_cap: numberInput("volatilityCap", "年化波动上限", .1, 100) / 100,
      risk_aversion: numberInput("riskAversion", "风险厌恶系数", .01, 100),
      range_utility_tolerance_bps: numberInput("rangeTolerance", "区间效用容忍度", 0, 100),
      no_trade_band: numberInput("noTradeBand", "主动调仓阈值", 0, 5) / 100,
      analysis_id: state.analysis?.analysis_id,
    };
    const account = numberInput("accountValue", "账户资金", 0.01, Number.MAX_SAFE_INTEGER, true);
    if (account !== undefined) payload.account_value = account;
    if (explicitCosts) {
      payload.cost_model = { commission_rate: numberInput("commissionRate", "佣金", 0, 10) / 100,
        transfer_rate: numberInput("transferRate", "过户费", 0, 10) / 100,
        buy_tax_rate: 0, sell_tax_rate: numberInput("sellTaxRate", "卖出税费", 0, 10) / 100,
        slippage_rate: numberInput("slippageRate", "滑点", 0, 10) / 100,
        minimum_commission: numberInput("minimumCommission", "最低佣金", 0, 10000) };
    }
    return payload;
  }

  function renderPlan(plan, input) {
    $("planEmpty").hidden = true;
    $("planResult").hidden = false;
    $("downloadPlan").disabled = false;
    $("cashWeight").textContent = percent(plan.cash_weight);
    $("stockBar").style.width = Math.max(0, Math.min(100, Number(plan.stock_weight || 0) * 100)) + "%";
    $("allocationLabel").textContent = "股票 " + percent(plan.stock_weight) + " / 现金 " + percent(plan.cash_weight);
    $("planIdentity").textContent = "收益／风险／成本优化 · " + state.snapshot.date + " 历史评分 · 名单 v" + plan.whitelist_version + " · " + roleLabel(plan.data_role);
    $("holdingSummary").textContent = input.holdings === null
      ? "未提供持仓，仅目标：不计算增减仓、换手或成交差额。"
      : "已提供当前持仓：计划差额 = 目标权重 − 当前权重。";
    $("executionSummary").textContent = input.assume_tradable
      ? "仅模拟：未明确阻塞的股票假设可交易；这不是实际成交确认。"
      : "未默认股票可交易：执行资料缺失时标记未知，只展示计划差额。";
    $("costSummary").textContent = input.holdings === null ? "换手与费用：未提供持仓，暂不估算。"
      : "预计换手 " + percent(plan.estimated_turnover) + " · 预计成本 " + percent(plan.estimated_cost)
        + " · 扣除估计费用后目标现金 " + percent(plan.cash_after_estimated_cost)
        + " · 考虑执行阻塞后模拟现金 " + percent(plan.projected_cash_weight);
    const explained = new Set([
      "OFFLINE_SIMULATION_ONLY", "SCORES_ARE_NOT_EXPECTED_RETURNS",
      "OFFLINE_DEMO_NOT_A_VALIDATED_STRATEGY", "HISTORICAL_REPLAY_NOT_LIVE_PREDICTION",
      "SYNTHETIC_DATA", "SYNTHETIC_ONLY", "SYNTHETIC_DATA_ONLY", "SYNTHETIC_DATA_NOT_MARKET_EVIDENCE",
      "离线配置模拟，未验证，不构成交易建议。",
      "得分仅用于排名；配置采用受约束等权，未拟合收益、协方差或风险最优权重。",
      "现金权重含预留费用；预计费用和成交模拟按调仓前账户净值计量。",
      "行业与可交易信息由本次模拟输入提供，未连接真实行情或订单系统。",
      "SELECTED_TARGETS_STRICTLY_POSITIVE", "RANKING_SCORE_DIFFERS_FROM_PORTFOLIO_UTILITY", "MISSING_HOLDINGS_TARGET_ONLY",
      "离线优化结果；工程复核不代表策略已通过真实市场回测。",
      "排名分数评价股票；目标效用比较整套权重，二者不是同一得分。",
      "建议区间固定其余目标、以现金吸收变化，限定效用损失；遵守最低目标与风险约束，不是置信区间，不能任意混合各股区间。",
      "推荐目标均为正；执行受阻、未交易和实际零持仓会另行标记。",
    ]);
    const notes = [...(plan.warnings || []), ...(plan.reason_codes || [])].filter((item) => !explained.has(item));
    warnings("planWarnings", notes);
    const obj = plan.objective;
    $("utilitySummary").textContent = plan.horizon + "日预期超额收益 " + percent(obj.expected_return)
      + " − 不确定性扣减 " + percent(obj.uncertainty_deduction) + " − 风险惩罚 " + percent(obj.risk_penalty)
      + " − 交易成本 " + percent(obj.transaction_cost) + " = 目标效用 " + percent(obj.utility)
      + (obj.transaction_cost === null ? "（持仓未知，暂未扣本次交易费）" : "");
    $("riskSummary").textContent = "预测年化波动 " + percent(obj.expected_volatility) + " / 上限 " + percent(plan.volatility_cap)
      + " · 行业权重：" + Object.entries(plan.industry_weights).map(([k, v]) => k + " " + percent(v)).join("；");
    $("stressSummary").textContent = "压力情景：成本×1.5 " + percent(plan.stress_checks.cost_1_5x)
      + "；最大行业 " + plan.stress_checks.largest_industry + " 同跌10%，组合损失 " + percent(plan.stress_checks.largest_industry_down_10pct_loss)
      + "；新增仓位比例往返成本 " + percent(plan.stress_checks.opening_round_trip_cost) + "（未含未来最低佣金）。"
      + "全部买入受阻时模拟现金 " + percent(plan.stress_checks.all_buys_blocked.cash_weight)
      + "；全部卖出受阻时模拟现金 " + percent(plan.stress_checks.all_sells_blocked.cash_weight) + "。";
    const statusText = { NO_TRADE: "保持原仓", NO_TRADE_REQUEST_UNSATISFIED: "费用不支持建仓，选择尚未执行",
      EXECUTION_INCOMPLETE: "目标有效，完整建仓尚未完成", OFFLINE_OPTIMIZED_NOT_VALIDATED: "离线目标已复核，策略尚未验证" };
    $("reviewSummary").textContent = "方案状态：" + (statusText[plan.status] || plan.status)
      + "。所选正目标与硬约束检查通过；真实市场回测复核尚未通过。";
    $("allocationReasons").replaceChildren(...(plan.allocation_diagnostics?.messages || []).map((text) => node("li", text)));
    $("intervalSummary").textContent = "建议区间：其余股票保持目标，以现金吸收单股变化；"
      + plan.horizon + "日模型效用最多下降 " + plan.range_utility_tolerance_bps + " 基点（"
      + percent(plan.range_utility_tolerance_bps / 10000) + " 账户净值）。容忍度是可调的工程设置，尚未通过真实回测验证。";
    const prior = state.previousPlan;
    const labels = {minimum_weight:"最低目标",market_budget:"预算上限",volatility_cap:"波动上限",
      risk_aversion:"风险厌恶系数",range_utility_tolerance_bps:"区间容忍度",no_trade_band:"调仓阈值",
      buy_cost:"买入费率",sell_cost:"卖出费率",cost_model:"分项费用",holdings:"当前持仓",
      account_value:"账户资金",assume_tradable:"可交易假设",analysis_id:"收益风险证据"};
    const comparable = prior && prior.snapshot_id === plan.snapshot_id && prior.whitelist_id === plan.whitelist_id
      && prior.whitelist_version === plan.whitelist_version;
    let comparison = "修改条件后点击生成，系统会重新求解，并与本页上一份相同名单的方案比较。";
    if (comparable) {
      const changes = Object.keys(labels).filter((key) => JSON.stringify(input[key]) !== JSON.stringify(state.previousInput[key]));
      const oldWeights = new Map(prior.positions.map((r) => [r.code, r]));
      const maxChange = Math.max(...plan.positions.map((r) => Math.abs(r.target_weight - oldWeights.get(r.code).target_weight)));
      const rangeChange = Math.max(...plan.positions.map((r) => Math.max(Math.abs(r.weight_lower-oldWeights.get(r.code).weight_lower), Math.abs(r.weight_upper-oldWeights.get(r.code).weight_upper))));
      comparison = "已重新求解。" + (changes.length ? "本次修改：" + changes.map((k) => labels[k]).join("、") + "。" : "输入参数与上一份相同。")
        + (maxChange < 1e-7 ? "目标权重未发生实质变化；请查看上述约束原因。" : "单股目标最大变化 " + percent(maxChange) + "（账户权重）。")
        + "区间边界最大变化 " + percent(rangeChange) + "；模型效用变化 " + percent(plan.objective.utility-prior.objective.utility, true) + "（参数口径可能不同）。";
    }
    $("parameterComparison").textContent = comparison;
    $("planAudit").textContent = JSON.stringify({ plan_id: plan.plan_id, analysis_id: plan.analysis_id,
      rule_version: plan.rule_version, snapshot_id: plan.snapshot_id, solver: plan.solver,
      cost_model: plan.cost_model, calibration: plan.calibration, risk_method: plan.risk_method,
      review: plan.review, allocation_diagnostics: plan.allocation_diagnostics,
      interval_method: plan.interval_method, range_utility_tolerance_bps: plan.range_utility_tolerance_bps,
      request_parameters: input, stress_checks: plan.stress_checks, orders_preview: plan.orders_preview }, null, 2);
    const fragment = document.createDocumentFragment();
    for (const row of plan.positions || []) {
      const tr = node("tr");
      const identity = node("td");
      identity.append(node("span", row.code + (row.selected ? "" : " · 原持仓"), "plan-code"),
        node("span", row.industry === "UNKNOWN" ? "未知行业" : row.industry || "未知行业", "plan-industry"));
      const targetCell = node("td", null, "numeric");
      targetCell.append(node("span", percent(row.target_weight), "plan-code"));
      const targetReason = row.reason_codes.includes("AT_STOCK_CAP") ? "已达单股上限"
        : row.reason_codes.includes("AT_USER_MINIMUM_WEIGHT") ? "已达最低目标" : "收益／风险／成本优化";
      targetCell.append(node("small", targetReason, "weight-reason"));
      const rangeCell = node("td", null, "numeric weight-range");
      const point = row.weight_upper-row.weight_lower < 1e-8;
      rangeCell.append(node("strong", point ? "固定 " + percent(row.target_weight) : percent(row.weight_lower) + "～" + percent(row.weight_upper)));
      if (point) rangeCell.append(node("small", ({FIXED_BY_BOUNDS:"上下限相同",FIXED_BY_DEADBAND:"调仓阈值固定",NO_FEASIBLE_WIDTH:"当前约束／容忍度下无可调空间"})[row.interval_status] || "无可调空间", "weight-reason"));
      tr.append(identity, node("td", percent(row.expected_return) + " / " + percent(row.return_standard_error), "numeric"),
        node("td", percent(row.current_weight), "numeric muted"), targetCell, rangeCell);
      const deltaClass = row.delta_weight > 0 ? "positive" : row.delta_weight < 0 ? "negative" : "muted";
      tr.append(node("td", percent(row.delta_weight, true), "numeric " + deltaClass), node("td", ACTIONS[row.action] || row.action || "—"));
      const execution = node("td", EXECUTION[row.execution_status] || row.execution_status || "执行信息未知",
        String(row.execution_status).includes("BLOCKED") ? "blocked" : "muted");
      execution.title = (row.reason_codes || []).map((code) => REASONS[code] || code).join("；");
      tr.append(execution, node("td", percent(row.executable_delta_weight, true), "numeric"),
        node("td", percent(row.projected_weight), "numeric"));
      const order = plan.orders_preview?.find((item) => item.code === row.code);
      tr.append(node("td", order ? (order.delta_quantity > 0 ? "+" : "") + order.delta_quantity : "—", "numeric"));
      fragment.append(tr);
    }
    $("planRows").replaceChildren(fragment);
    $("forwardEvents").replaceChildren();
    $("eventSummary").textContent = "读取这份方案的后续观察…";
    void loadEvents(plan.plan_id);
  }

  async function generatePlan() {
    if (state.generating || state.loading) return;
    let input;
    try { input = planPayload(); } catch (error) { notify(error.message, "error"); return; }
    const epoch = state.epoch, edit = state.edit;
    state.plan = null;
    $("planResult").hidden = true;
    $("planEmpty").hidden = false;
    $("downloadPlan").disabled = true;
    state.generating = true;
    updateControls();
    try {
      const plan = await request("/plans", { method: "POST", body: JSON.stringify(input) });
      if (epoch !== state.epoch || edit !== state.edit) return;
      if (plan.snapshot_id !== state.snapshot.snapshot_id || plan.whitelist_id !== input.whitelist_id
          || plan.whitelist_version !== input.whitelist_version) throw new Error("返回方案与当前快照或名单版本不一致，已拒绝展示，请重新生成。");
      state.plan = plan;
      renderPlan(plan, input);
      state.previousPlan = plan;
      state.previousInput = input;
      notify("离线配置模拟已生成。请核对现金、目标比例及执行阻塞。", "success");
      $("planTitle").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) { if (epoch === state.epoch && edit === state.edit) notify(error.message, "error"); }
    finally { state.generating = false; updateControls(); }
  }

  async function loadEvents(planId) {
    try {
      const result = await request("/plans/" + encodeURIComponent(planId) + "/events");
      if (state.plan?.plan_id !== planId) return;
      $("forwardEvents").replaceChildren(...result.events.map((event) => node("li",
        event.observed_at + " · " + ({adopted:"采纳",declined:"未采纳",execution:"执行观察",valuation:"账户净值",note:"备注"}[event.kind])
        + (event.account_equity !== null && event.account_equity !== undefined ? " " + event.account_equity : "") + " · " + event.note)));
      $("eventSummary").textContent = "已保存 " + result.events.length + " 条记录。" + result.performance_note;
    } catch (error) { if (state.plan?.plan_id === planId) $("eventSummary").textContent = error.message; }
  }

  async function recordEvent() {
    const planId = state.plan?.plan_id;
    if (!planId || $("recordEvent").disabled) return;
    $("recordEvent").disabled = true;
    try {
      const kind = $("eventKind").value;
      const payload = { kind, observed_at: new Date().toISOString(), note: $("eventNote").value,
        account_equity: kind === "valuation" ? numberInput("eventEquity", "账户净值", .01, Number.MAX_SAFE_INTEGER) : null };
      await request("/plans/" + encodeURIComponent(planId) + "/events", { method: "POST", body: JSON.stringify(payload) });
      await loadEvents(planId);
    } catch (error) { notify(error.message, "error"); }
    finally { $("recordEvent").disabled = false; }
  }

  async function importBundle() {
    const path = $("manifestPath").value.trim();
    if (!path) { notify("请填写本机数据包的 manifest 文件路径。", "error"); return; }
    if (state.importing) return;
    state.importing = true;
    updateControls();
    try {
      const bundle = await request("/bundles/import", { method: "POST", body: JSON.stringify({ manifest_path: path }) });
      const id = await listBundles(bundle.bundle_id);
      if (id) await loadBundle(id);
      notify("本地数据包已导入。请核对实际股票数量和数据类型。", "success");
    } catch (error) { notify(error.message, "error"); }
    finally { state.importing = false; updateControls(); }
  }

  function downloadPlan() {
    if (!state.plan) return;
    const blob = new Blob([JSON.stringify(state.plan, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = node("a");
    link.href = url;
    link.download = "wingman-offline-plan-" + String(state.snapshot.date).replace(/[^0-9-]/g, "") + ".json";
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  $("bundleSelect").addEventListener("change", () => { void loadBundle($("bundleSelect").value); });
  $("dateSelect").addEventListener("change", () => { fillFolds(); void loadScores(); });
  $("foldSelect").addEventListener("change", () => { void loadScores(); });
  $("reloadScores").addEventListener("click", () => { void loadScores(); });
  $("stockSearch").addEventListener("input", renderStocks);
  $("selectVisible").addEventListener("click", () => {
    for (const row of visibleRows()) state.selected.add(row.code);
    invalidatePlan("股票选择已更改，请保存白名单后重新生成方案。");
    renderStocks();
  });
  $("clearSelection").addEventListener("click", () => {
    state.selected.clear();
    invalidatePlan("股票选择已更改，请保存白名单后重新生成方案。");
    renderStocks();
  });
  $("whitelistName").addEventListener("input", () => invalidatePlan("白名单名称已更改，请保存后重新生成方案。"));
  $("saveWhitelist").addEventListener("click", () => { void saveWhitelist(); });
  $("saveAsNew").addEventListener("click", () => { void saveWhitelist(true); });
  $("savedWhitelistSelect").addEventListener("change", updateControls);
  $("restoreWhitelist").addEventListener("click", () => { void restoreWhitelist(); });
  $("holdingsMode").addEventListener("change", () => {
    $("holdingsEditor").hidden = $("holdingsMode").value !== "provided";
    invalidatePlan("当前持仓条件已更改，请重新生成方案。");
  });
  for (const id of ["marketBudget", "accountValue", "holdingsText", "buyCost", "sellCost", "minimumWeight", "volatilityCap",
    "riskAversion", "rangeTolerance", "noTradeBand", "commissionRate", "transferRate", "sellTaxRate", "slippageRate", "minimumCommission"]) {
    $(id).addEventListener("input", () => invalidatePlan("模拟参数已更改，请重新生成方案。"));
  }
  $("assumeTradable").addEventListener("change", () => invalidatePlan("执行假设已更改，请重新生成方案。"));
  $("generatePlan").addEventListener("click", () => { void generatePlan(); });
  $("importBundle").addEventListener("click", () => { void importBundle(); });
  $("downloadPlan").addEventListener("click", downloadPlan);
  $("recordEvent").addEventListener("click", () => { void recordEvent(); });
  $("explicitCosts").addEventListener("change", () => {
    $("costComponents").hidden = !$("explicitCosts").checked;
    $("buyCost").disabled = $("sellCost").disabled = $("explicitCosts").checked;
    invalidatePlan("成本模型已更改，请重新生成方案。");
  });
  $("importAnalysis").addEventListener("click", async () => {
    if (!state.snapshot || $("importAnalysis").disabled) return;
    const epoch = state.epoch, snapshotId = state.snapshot.snapshot_id;
    $("importAnalysis").disabled = true;
    try {
      await request("/analysis/import", { method: "POST", body: JSON.stringify({ snapshot_id: snapshotId,
        manifest_path: $("analysisPath").value.trim() }) });
      if (epoch === state.epoch) { invalidatePlan("配置证据已更新，请重新生成方案。"); await loadAnalysis(snapshotId, epoch); }
    } catch (error) { if (epoch === state.epoch) notify(error.message, "error"); }
    finally { $("importAnalysis").disabled = false; }
  });

  async function initialize() {
    updateControls();
    const results = await Promise.allSettled([listBundles(), listWhitelists()]);
    if (results[0].status === "fulfilled" && results[0].value) await loadBundle(results[0].value);
    else if (results[0].status === "rejected") {
      setOptions($("bundleSelect"), [], "数据包加载失败，可尝试导入");
      notify(results[0].reason.message, "error");
    } else {
      $("snapshotMeta").textContent = "尚无数据包。请展开「导入本地研究数据包」添加已保存的评分。";
    }
    if (results[1].status === "rejected") notify("白名单列表加载失败：" + results[1].reason.message, "error");
  }

  void initialize();
})();
