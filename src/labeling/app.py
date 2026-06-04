# -*- coding: utf-8 -*-
"""网页版 patch 级盐碱地标注工具 (Flask 后端 + 内嵌单页前端)。

数据流: 浏览器 <-JSON-> Flask <-文件IO-> data/labels/labels_v1.csv

启动:
  1. 读 data/labels/indices_summary.csv (已按 salinity_prior 降序)
  2. 读 data/labels/labels_v1.csv (不存在则建空模板)
  3. 定位第一个未标注 patch 作为起点

用法: conda run -n saline python src/labeling/app.py
然后浏览器打开 http://localhost:5000

标注顺序 = salinity_prior 降序 (高先验的先看)。最终标签以人工判断为准。
"""
import sys
import csv
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, Response

from config import DATA_LABELS

INDICES_CSV = DATA_LABELS / "indices_summary.csv"
LABELS_CSV = DATA_LABELS / "labels_v1.csv"
PREVIEW_DIR = DATA_LABELS / "preview_images"

LABEL_FIELDS = ["patch_id", "label", "confidence", "notes", "labeled_at"]

app = Flask(__name__)

# ---- 内存数据 ----
PATCHES = []        # 按 salinity_prior 降序的 patch 元信息 (list of dict)
ORDER = []          # patch_id 顺序
LABELS = {}         # patch_id -> {label, confidence, notes, labeled_at}


def load_patches() -> None:
    """读 indices_summary.csv -> PATCHES (保持文件内顺序, 即 prior 降序)。"""
    df = pd.read_csv(INDICES_CSV)
    PATCHES.clear()
    ORDER.clear()
    for _, r in df.iterrows():
        tags = [
            t for t, on in (
                ("water_dominant", bool(r["water_dominant"])),
                ("vegetation_dominant", bool(r["vegetation_dominant"])),
                ("bare_soil_dominant", bool(r["bare_soil_dominant"])),
            ) if on
        ]
        PATCHES.append({
            "patch_id": str(r["patch_id"]),
            "region": str(r["region"]),
            "salinity_prior": float(r["salinity_prior"]),
            "rank_overall": int(r["rank_overall"]),
            "rank_in_region": int(r["rank_in_region"]),
            "ndvi_mean": float(r["ndvi_mean"]),
            "ndwi_mean": float(r["ndwi_mean"]),
            "si_mean": float(r["si_mean"]),
            "ndsi_swir_mean": float(r["ndsi_swir_mean"]),
            "sr_swir_mean": float(r["sr_swir_mean"]),
            "vh_vv_ratio_mean": float(r["vh_vv_ratio_mean"]),
            "vv_mean": float(r["vv_mean"]),
            "tags": tags,
        })
        ORDER.append(str(r["patch_id"]))


def load_labels() -> None:
    """读 labels_v1.csv (不存在则建空模板)。"""
    LABELS.clear()
    if not LABELS_CSV.exists():
        with open(LABELS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=LABEL_FIELDS).writeheader()
        return
    df = pd.read_csv(LABELS_CSV, dtype=str).fillna("")
    for _, r in df.iterrows():
        pid = r["patch_id"]
        if pid:
            LABELS[pid] = {
                "label": r.get("label", ""),
                "confidence": r.get("confidence", ""),
                "notes": r.get("notes", ""),
                "labeled_at": r.get("labeled_at", ""),
            }


def save_labels() -> None:
    """把 LABELS 全量写回 CSV (60 行, 简单稳妥)。"""
    with open(LABELS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for pid in ORDER:
            if pid in LABELS:
                row = {"patch_id": pid, **LABELS[pid]}
                w.writerow(row)


def first_unlabeled_index() -> int:
    for i, pid in enumerate(ORDER):
        if pid not in LABELS or LABELS[pid].get("label", "") == "":
            return i
    return 0  # 全部标完, 回到第一个


# ---------------- 路由 ----------------
@app.route("/")
def index() -> Response:
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/state")
def api_state():
    labeled = {pid: LABELS[pid] for pid in ORDER if pid in LABELS and LABELS[pid].get("label", "") != ""}
    return jsonify({
        "total": len(ORDER),
        "labeled_count": len(labeled),
        "order": ORDER,
        "labels": labeled,
        "start_index": first_unlabeled_index(),
    })


@app.route("/api/patch/<patch_id>")
def api_patch(patch_id: str):
    for p in PATCHES:
        if p["patch_id"] == patch_id:
            existing = LABELS.get(patch_id, {})
            return jsonify({**p, "existing": existing})
    return jsonify({"error": "patch not found"}), 404


@app.route("/preview/<path:filename>")
def preview(filename: str):
    return send_from_directory(PREVIEW_DIR, filename)


@app.route("/api/label", methods=["POST"])
def api_label():
    data = request.get_json(force=True)
    pid = data.get("patch_id", "")
    if pid not in ORDER:
        return jsonify({"error": "unknown patch_id"}), 400
    LABELS[pid] = {
        "label": str(data.get("label", "")),
        "confidence": str(data.get("confidence", "")),
        "notes": str(data.get("notes", "")),
        "labeled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_labels()
    labeled_count = sum(
        1 for x in ORDER if x in LABELS and LABELS[x].get("label", "") != ""
    )
    return jsonify({"ok": True, "labeled_count": labeled_count, "total": len(ORDER)})


# ---------------- 前端 (单页) ----------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>盐碱地标注工具</title>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-100 text-slate-800">
<div id="app" class="max-w-6xl mx-auto p-4">

  <!-- 头部 -->
  <div class="bg-white rounded-xl shadow p-4 mb-4">
    <div class="flex items-center justify-between">
      <h1 class="text-xl font-bold text-indigo-700">盐碱地标注工具</h1>
      <div class="text-sm text-slate-500">快捷键: 1 / 0 / -(skip) · H/M/L · Enter 保存下一个 · ←/→ 翻页</div>
    </div>
    <div class="mt-3">
      <div class="flex items-center gap-3">
        <div class="flex-1 bg-slate-200 rounded-full h-3 overflow-hidden">
          <div id="bar" class="bg-indigo-600 h-3" style="width:0%"></div>
        </div>
        <div id="progress" class="text-sm font-semibold whitespace-nowrap">0/0</div>
      </div>
      <div class="mt-2 text-sm text-slate-600">
        Current: <span id="cur-id" class="font-mono font-semibold"></span>
        · region: <span id="cur-region"></span>
        · rank <span id="cur-rank"></span>
        · salinity_prior: <span id="cur-prior" class="font-semibold text-indigo-700"></span>
      </div>
    </div>
  </div>

  <!-- 双图 -->
  <div class="grid grid-cols-2 gap-4 mb-4">
    <div class="bg-white rounded-xl shadow p-3">
      <div class="text-center text-sm font-semibold text-slate-600 mb-2">RGB 真彩色 (B4/B3/B2)</div>
      <img id="img-rgb" class="w-full rounded-lg bg-black" alt="rgb">
    </div>
    <div class="bg-white rounded-xl shadow p-3">
      <div class="text-center text-sm font-semibold text-slate-600 mb-2">SWIR 假彩色 (B11/B8/B4)</div>
      <img id="img-swir" class="w-full rounded-lg bg-black" alt="swir">
    </div>
  </div>

  <!-- 指数 -->
  <div class="bg-white rounded-xl shadow p-4 mb-4">
    <div class="text-sm font-semibold text-slate-600 mb-2">Key indices</div>
    <div class="grid grid-cols-3 sm:grid-cols-4 gap-2 text-sm font-mono" id="indices"></div>
    <div class="mt-2 text-sm">Tags: <span id="tags" class="font-semibold text-amber-700"></span></div>
  </div>

  <!-- 标注 -->
  <div class="bg-white rounded-xl shadow p-4">
    <div class="text-sm font-semibold text-slate-600 mb-2">Label this patch</div>
    <div class="grid grid-cols-3 gap-3 mb-4">
      <button data-label="1" class="lbl-btn py-4 rounded-xl text-lg font-bold border-2 border-rose-300 bg-rose-50 hover:bg-rose-100">1 · Saline</button>
      <button data-label="0" class="lbl-btn py-4 rounded-xl text-lg font-bold border-2 border-emerald-300 bg-emerald-50 hover:bg-emerald-100">0 · Non-saline</button>
      <button data-label="-1" class="lbl-btn py-4 rounded-xl text-lg font-bold border-2 border-slate-300 bg-slate-50 hover:bg-slate-100">- · Skip</button>
    </div>

    <div class="flex items-center gap-4 mb-3">
      <span class="text-sm font-semibold text-slate-600">Confidence:</span>
      <label class="conf-lbl cursor-pointer px-3 py-1 rounded-lg border" data-conf="high">H · high</label>
      <label class="conf-lbl cursor-pointer px-3 py-1 rounded-lg border" data-conf="medium">M · medium</label>
      <label class="conf-lbl cursor-pointer px-3 py-1 rounded-lg border" data-conf="low">L · low</label>
    </div>

    <div class="mb-4">
      <input id="notes" type="text" placeholder="Notes (可选)"
             class="w-full border rounded-lg px-3 py-2 text-sm">
    </div>

    <div class="flex gap-3">
      <button id="prev" class="px-4 py-3 rounded-xl border-2 border-slate-300 bg-white hover:bg-slate-50 font-semibold">← Previous</button>
      <button id="save" class="flex-1 px-4 py-3 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 font-bold text-lg">Save &amp; Next ↵</button>
      <button id="next" class="px-4 py-3 rounded-xl border-2 border-slate-300 bg-white hover:bg-slate-50 font-semibold">Skip → </button>
    </div>
    <div id="status" class="mt-3 text-sm text-slate-500"></div>
  </div>
</div>

<!-- 完成页 -->
<div id="done" class="hidden fixed inset-0 bg-indigo-700 text-white flex-col items-center justify-center text-center p-8" style="display:none">
  <div class="text-6xl mb-4">🎉</div>
  <div class="text-3xl font-bold mb-2">完成!</div>
  <div id="done-msg" class="text-xl">60/60 已标注</div>
  <div class="mt-4 text-indigo-200 text-sm">可以关闭 Flask, 然后验证 labels_v1.csv。</div>
</div>

<script>
let ORDER = [], LABELS = {}, idx = 0, TOTAL = 0;
let sel = { label: null, confidence: null };

async function loadState() {
  const s = await (await fetch('/api/state')).json();
  ORDER = s.order; LABELS = s.labels; TOTAL = s.total;
  idx = s.start_index;
  updateProgress();
  await showPatch();
}

function updateProgress() {
  const n = Object.keys(LABELS).length;
  document.getElementById('progress').textContent = n + '/' + TOTAL;
  document.getElementById('bar').style.width = (TOTAL ? (n/TOTAL*100) : 0) + '%';
}

function setLabelSel(v) {
  sel.label = v;
  document.querySelectorAll('.lbl-btn').forEach(b => {
    const on = b.dataset.label === String(v);
    b.classList.toggle('ring-4', on);
    b.classList.toggle('ring-indigo-400', on);
  });
}
function setConfSel(v) {
  sel.confidence = v;
  document.querySelectorAll('.conf-lbl').forEach(b => {
    const on = b.dataset.conf === v;
    b.classList.toggle('bg-indigo-600', on);
    b.classList.toggle('text-white', on);
  });
}

async function showPatch() {
  if (idx < 0) idx = 0;
  if (idx >= TOTAL) idx = TOTAL - 1;
  const pid = ORDER[idx];
  const p = await (await fetch('/api/patch/' + pid)).json();

  document.getElementById('cur-id').textContent = p.patch_id;
  document.getElementById('cur-region').textContent = p.region;
  document.getElementById('cur-rank').textContent = '#' + p.rank_overall + '/' + TOTAL;
  document.getElementById('cur-prior').textContent = p.salinity_prior.toFixed(4);

  document.getElementById('img-rgb').src = '/preview/' + p.patch_id + '_rgb.png';
  document.getElementById('img-swir').src = '/preview/' + p.patch_id + '_swir.png';

  const fmt = (k, v) => '<div><span class="text-slate-400">'+k+':</span> '+v+'</div>';
  document.getElementById('indices').innerHTML =
    fmt('NDVI', p.ndvi_mean.toFixed(3)) +
    fmt('NDWI', p.ndwi_mean.toFixed(3)) +
    fmt('SI', p.si_mean.toFixed(1)) +
    fmt('NDSI-SWIR', p.ndsi_swir_mean.toFixed(3)) +
    fmt('SR-SWIR', p.sr_swir_mean.toFixed(3)) +
    fmt('VV', p.vv_mean.toFixed(2)) +
    fmt('VH/VV', p.vh_vv_ratio_mean.toFixed(3));
  document.getElementById('tags').textContent = p.tags.length ? p.tags.join(', ') : '(none)';

  // 预填已有标注
  const ex = p.existing || {};
  setLabelSel(ex.label !== undefined && ex.label !== '' ? ex.label : null);
  setConfSel(ex.confidence || null);
  document.getElementById('notes').value = ex.notes || '';
  document.getElementById('status').textContent =
    ex.label ? ('已标注: label='+ex.label+' ('+ (ex.confidence||'?') +') @ '+(ex.labeled_at||'')) : '未标注';
}

async function saveAndNext() {
  const pid = ORDER[idx];
  if (sel.label === null) { flash('请先选 label (1/0/-)'); return; }
  const body = {
    patch_id: pid, label: sel.label,
    confidence: sel.confidence || '', notes: document.getElementById('notes').value
  };
  const res = await (await fetch('/api/label', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body)
  })).json();
  LABELS[pid] = { label: String(sel.label), confidence: body.confidence, notes: body.notes };
  updateProgress();

  if (res.labeled_count >= res.total) { showDone(res.labeled_count, res.total); return; }
  // 跳到下一个未标注; 若没有则下一个
  let ni = idx + 1;
  while (ni < TOTAL && ORDER[ni] in LABELS) ni++;
  idx = (ni < TOTAL) ? ni : Math.min(idx + 1, TOTAL - 1);
  showPatch();
}

function showDone(n, t) {
  const d = document.getElementById('done');
  d.style.display = 'flex';
  document.getElementById('done-msg').textContent = n + '/' + t + ' 已标注';
}

function flash(msg) {
  const s = document.getElementById('status');
  s.textContent = msg; s.classList.add('text-rose-600');
  setTimeout(() => s.classList.remove('text-rose-600'), 1200);
}

// 按钮
document.querySelectorAll('.lbl-btn').forEach(b =>
  b.addEventListener('click', () => setLabelSel(b.dataset.label)));
document.querySelectorAll('.conf-lbl').forEach(b =>
  b.addEventListener('click', () => setConfSel(b.dataset.conf)));
document.getElementById('save').addEventListener('click', saveAndNext);
document.getElementById('prev').addEventListener('click', () => { idx--; showPatch(); });
document.getElementById('next').addEventListener('click', () => { idx++; showPatch(); });

// 键盘
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.key !== 'Enter') return;
  switch (e.key) {
    case '1': setLabelSel('1'); break;
    case '0': setLabelSel('0'); break;
    case '-': setLabelSel('-1'); break;
    case 'h': case 'H': setConfSel('high'); break;
    case 'm': case 'M': setConfSel('medium'); break;
    case 'l': case 'L': setConfSel('low'); break;
    case 'Enter': saveAndNext(); break;
    case 'ArrowRight': idx++; showPatch(); break;
    case 'ArrowLeft': idx--; showPatch(); break;
  }
});

loadState();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    load_patches()
    load_labels()
    print(f"已加载 {len(PATCHES)} 个 patch, 已标注 {sum(1 for p in ORDER if p in LABELS and LABELS[p].get('label'))} 个")
    print("打开 http://localhost:5000 开始标注")
    app.run(host="127.0.0.1", port=5000, debug=False)
