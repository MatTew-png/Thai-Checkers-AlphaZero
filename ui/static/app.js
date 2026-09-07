// Thai-Checkers-AlphaZero Frontend Logic

let gameState = null;
let selectedSq = null;
let humanSide = 1; // 1 = White, -1 = Black
let soundEnabled = true;
let isAutoPlaying = false;
let isAIThinking = false;

// Audio Synthesizer via Web Audio API
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

function playSound(type) {
  if (!soundEnabled) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);

    if (type === 'move') {
      osc.frequency.setValueAtTime(320, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(160, audioCtx.currentTime + 0.08);
      gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
      gain.gain.linearRampToValueAtTime(0.01, audioCtx.currentTime + 0.08);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.08);
    } else if (type === 'capture') {
      osc.frequency.setValueAtTime(520, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(120, audioCtx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.5, audioCtx.currentTime);
      gain.gain.linearRampToValueAtTime(0.01, audioCtx.currentTime + 0.15);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.15);
    } else if (type === 'king') {
      osc.frequency.setValueAtTime(440, audioCtx.currentTime);
      osc.frequency.setValueAtTime(660, audioCtx.currentTime + 0.1);
      gain.gain.setValueAtTime(0.4, audioCtx.currentTime);
      gain.gain.linearRampToValueAtTime(0.01, audioCtx.currentTime + 0.25);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.25);
    }
  } catch (e) {
    console.error(e);
  }
}

// Mapping between 32 playable squares and (r, c)
const FILES = 'ABCDEFGH';
const RANKS = '12345678';
const SQ_TO_COORD = [];
const COORD_TO_SQ = {};
let sqCount = 0;
for (let r = 0; r < 8; r++) {
  for (let c = 0; c < 8; c++) {
    if ((r + c) % 2 === 0) {
      SQ_TO_COORD.push([r, c]);
      COORD_TO_SQ[`${r},${c}`] = sqCount;
      sqCount++;
    }
  }
}

function coordToAlg(r, c) {
  return `${FILES[c]}${RANKS[r]}`;
}

function sqToAlg(sq) {
  if (sq === null || sq === undefined || sq < 0 || sq >= 32) return '-';
  const [r, c] = SQ_TO_COORD[sq];
  return coordToAlg(r, c);
}

// Initialize Board DOM
const boardEl = document.getElementById('board');

function createBoardDOM() {
  boardEl.innerHTML = '';
  // Row 7 (top) down to Row 0 (bottom)
  for (let r = 7; r >= 0; r--) {
    for (let c = 0; c < 8; c++) {
      const cell = document.createElement('div');
      cell.className = 'board-cell';
      cell.dataset.r = r;
      cell.dataset.c = c;

      const isPlayable = (r + c) % 2 === 0;
      if (isPlayable) {
        cell.style.backgroundColor = '#543d2b';
        cell.dataset.sq = COORD_TO_SQ[`${r},${c}`];
      } else {
        cell.style.backgroundColor = '#d6c5a5';
      }

      cell.addEventListener('click', () => onCellClick(r, c));
      boardEl.appendChild(cell);
    }
  }
}

// Fetch Board State from Server
async function fetchState() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    gameState = data;
    renderBoard();
    updateUI();
    checkAITurn();
  } catch (err) {
    console.error('Failed to fetch state:', err);
  }
}

// Reset Game
async function resetGame() {
  isAutoPlaying = false;
  isAIThinking = false;
  selectedSq = null;
  document.getElementById('btn-auto-play').innerText = '▶️ AI ดวล AI';
  document.getElementById('move-history').innerHTML = '<p class="text-slate-500 italic">เริ่มต้นเกม...</p>';

  try {
    const res = await fetch('/api/reset', { method: 'POST' });
    gameState = await res.json();
    renderBoard();
    updateUI();
    checkAITurn();
  } catch (err) {
    console.error('Failed to reset:', err);
  }
}

// Render Pieces & Highlights
function renderBoard() {
  if (!gameState) return;

  // Clear previous piece elements and highlights
  document.querySelectorAll('.board-cell').forEach(cell => {
    cell.innerHTML = '';
    cell.classList.remove('selected-sq', 'valid-dest', 'forced-indicator');
  });

  const legalMoves = gameState.legal_moves || [];
  const validDests = selectedSq !== null
    ? legalMoves.filter(m => m.from_sq === selectedSq).map(m => m.to_sq)
    : [];

  const forcedSquares = new Set(
    gameState.has_forced_capture ? legalMoves.map(m => m.from_sq) : []
  );

  for (let sq = 0; sq < 32; sq++) {
    const piece = gameState.squares[sq];
    const [r, c] = SQ_TO_COORD[sq];
    const cell = document.querySelector(`.board-cell[data-r="${r}"][data-c="${c}"]`);
    if (!cell) continue;

    // Highlight selected square
    if (selectedSq === sq) {
      cell.classList.add('selected-sq');
    }

    // Highlight valid destinations
    if (validDests.includes(sq)) {
      cell.classList.add('valid-dest');
    }

    // Highlight pieces with forced capture
    if (forcedSquares.has(sq) && gameState.current_player === humanSide) {
      cell.classList.add('forced-indicator');
    }

    // Render piece
    if (piece !== 0) {
      const pEl = document.createElement('div');
      pEl.className = `piece ${piece > 0 ? 'piece-p1' : 'piece-p2'}`;

      if (Math.abs(piece) === 2) {
        // King crown
        pEl.innerHTML = '<span class="king-badge">👑</span>';
      }

      cell.appendChild(pEl);
    }
  }
}

// Cell Click Handler
async function onCellClick(r, c) {
  if (gameState.is_game_over || isAIThinking) return;

  const coordKey = `${r},${c}`;
  if (!(coordKey in COORD_TO_SQ)) return; // light square

  const clickedSq = COORD_TO_SQ[coordKey];
  const piece = gameState.squares[clickedSq];
  const isMyPiece = (humanSide === 1 && piece > 0) || (humanSide === -1 && piece < 0);

  // If a piece was already selected and user clicked a valid destination
  if (selectedSq !== null) {
    const legalMoves = gameState.legal_moves || [];
    const move = legalMoves.find(m => m.from_sq === selectedSq && m.to_sq === clickedSq);

    if (move) {
      // Execute human move!
      selectedSq = null;
      renderBoard();
      await executeHumanMove(move);
      return;
    }
  }

  // Select piece if it belongs to active player
  if (gameState.current_player === humanSide && isMyPiece) {
    // If mid-multi-jump, can only select continuation piece
    if (gameState.continuation_sq !== null && clickedSq !== gameState.continuation_sq) {
      return;
    }
    // If forced capture exists, can only select pieces that have captures
    if (gameState.has_forced_capture) {
      const hasCap = gameState.legal_moves.some(m => m.from_sq === clickedSq);
      if (!hasCap) return;
    }

    selectedSq = clickedSq;
    renderBoard();
  } else {
    selectedSq = null;
    renderBoard();
  }
}

// Execute Human Move API Call
async function executeHumanMove(move) {
  playSound(move.is_capture ? 'capture' : (move.promoted ? 'king' : 'move'));

  try {
    const res = await fetch('/api/human_move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ from_sq: move.from_sq, to_sq: move.to_sq })
    });

    if (res.ok) {
      gameState = await res.json();
      logMove('คุณ', move);
      renderBoard();
      updateUI();
      checkAITurn();
    }
  } catch (err) {
    console.error('Move error:', err);
  }
}

// AI Turn Execution
async function checkAITurn() {
  if (gameState.is_game_over || isAIThinking) return;

  const isAITurn = (gameState.current_player !== humanSide) || isAutoPlaying;
  if (!isAITurn) return;

  await triggerAIMove();
}

async function triggerAIMove() {
  if (gameState.is_game_over || isAIThinking) return;
  isAIThinking = true;

  const diff = document.getElementById('select-difficulty').value;
  let sims = 150;
  let engine = 'alphazero';

  if (diff === 'az-50') sims = 50;
  else if (diff === 'az-150') sims = 150;
  else if (diff === 'az-400') sims = 400;
  else if (diff === 'minimax') engine = 'minimax';

  document.getElementById('turn-text').innerHTML = '🤖 <span class="animate-pulse">AlphaZero กำลังคำนวณการเดิน...</span>';

  try {
    const res = await fetch('/api/ai_move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ simulations: sims, engine: engine, temperature: 0.0 })
    });

    if (res.ok) {
      const data = await res.json();
      gameState = data;

      if (data.ai_move) {
        playSound('move');
        logMove(data.current_player === 1 ? 'AI 1' : 'AlphaZero', data.ai_move);
      }

      if (data.mcts_candidates) {
        renderMCTSStream(data.mcts_candidates);
      }

      renderBoard();
      updateUI();
    }
  } catch (err) {
    console.error('AI move error:', err);
  } finally {
    isAIThinking = false;
    if (isAutoPlaying && !gameState.is_game_over) {
      setTimeout(triggerAIMove, 600);
    }
  }
}

// Update UI Indicators
function updateUI() {
  if (!gameState) return;

  // Status Banner
  const turnDot = document.getElementById('turn-dot');
  const turnText = document.getElementById('turn-text');
  const forcedAlert = document.getElementById('forced-alert');

  if (gameState.is_game_over) {
    turnDot.className = 'w-3 h-3 rounded-full bg-red-500';
    if (gameState.winner === 0) {
      turnText.innerText = '🤝 ผลการแข่งขัน: เสมอกัน (Draw)';
    } else if (gameState.winner === humanSide) {
      turnText.innerText = '🎉 ยินดีด้วย! คุณเอาชนะ AI สำเร็จ (Human Wins)';
    } else {
      turnText.innerText = '🤖 AlphaZero ชนะการแข่งขัน (AI Wins)';
    }
    forcedAlert.classList.add('hidden');
  } else {
    const isP1 = gameState.current_player === 1;
    turnDot.className = `w-3 h-3 rounded-full ${isP1 ? 'bg-amber-400' : 'bg-slate-400'} animate-pulse`;
    turnText.innerText = `ตาเดิน: ${isP1 ? 'ผู้เล่น 1 (สีขาว)' : 'ผู้เล่น 2 (สีดำ)'}`;

    if (gameState.has_forced_capture) {
      forcedAlert.classList.remove('hidden');
    } else {
      forcedAlert.classList.add('hidden');
    }
  }

  // Evaluation Bar
  const evalScore = gameState.evaluation_p1 || 0.0;
  document.getElementById('eval-score').innerText = `${evalScore >= 0 ? '+' : ''}${evalScore.toFixed(2)}`;

  // Convert V in [-1, 1] to P1 percentage [0..100%]
  const p1Pct = Math.round(((evalScore + 1) / 2) * 100);
  const p2Pct = 100 - p1Pct;

  document.getElementById('bar-p1').style.width = `${p1Pct}%`;
  document.getElementById('bar-p2').style.width = `${p2Pct}%`;
  document.getElementById('pct-p1').innerText = `ผู้เล่น 1 (ขาว): ${p1Pct}%`;
  document.getElementById('pct-p2').innerText = `ผู้เล่น 2 (ดำ): ${p2Pct}%`;

  // Plies counter
  document.getElementById('ply-counter').innerText = `${gameState.ply_count} Plies (Clock: ${gameState.halfmove_clock})`;
}

// Render MCTS Candidate Thoughts
function renderMCTSStream(candidates) {
  const container = document.getElementById('mcts-candidates');
  if (!candidates || candidates.length === 0) {
    container.innerHTML = '<p class="text-slate-500 italic">ไม่มีข้อมูลการสำรวจ</p>';
    return;
  }

  let html = '';
  candidates.forEach((c, idx) => {
    const fromAlg = c.from_algebraic || sqToAlg(c.from_sq);
    const toAlg = c.to_algebraic || sqToAlg(c.to_sq);
    const notation = c.notation || `${fromAlg} ➔ ${toAlg}`;
    html += `
      <div class="flex items-center justify-between p-2 rounded-lg bg-slate-800/80 border border-slate-700">
        <div class="flex items-center space-x-2">
          <span class="font-mono text-amber-400 font-bold">#${idx + 1}</span>
          <span class="font-mono font-bold text-slate-100">${notation}</span>
          <span class="text-slate-500 text-[10px] font-mono">(${fromAlg}➔${toAlg})</span>
        </div>
        <div class="flex items-center space-x-3">
          <span class="text-slate-400 font-mono text-[11px]">${c.visits} visits</span>
          <span class="px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 font-mono font-bold">${c.percentage.toFixed(1)}%</span>
        </div>
      </div>
    `;
  });
  container.innerHTML = html;
}

// Log Move History
function logMove(player, move) {
  const container = document.getElementById('move-history');
  if (gameState.ply_count === 1) {
    container.innerHTML = '';
  }

  const fromAlg = move.from_algebraic || sqToAlg(move.from_sq);
  const toAlg = move.to_algebraic || sqToAlg(move.to_sq);
  const sep = move.is_capture ? 'x' : '➔';
  const notation = move.notation || `${fromAlg} ${sep} ${toAlg}`;

  const pEl = document.createElement('div');
  pEl.className = 'flex justify-between py-0.5 border-b border-slate-800/50';
  pEl.innerHTML = `
    <span class="text-slate-400 font-semibold">${player}:</span>
    <span class="font-mono font-bold text-amber-300">${notation}</span>
  `;
  container.appendChild(pEl);
  container.scrollTop = container.scrollHeight;
}

// Setup Event Listeners
document.getElementById('btn-new-game').addEventListener('click', resetGame);
document.getElementById('btn-ai-move').addEventListener('click', triggerAIMove);

document.getElementById('btn-auto-play').addEventListener('click', () => {
  isAutoPlaying = !isAutoPlaying;
  document.getElementById('btn-auto-play').innerText = isAutoPlaying ? '⏸️ หยุดดวล' : '▶️ AI ดวล AI';
  if (isAutoPlaying) {
    triggerAIMove();
  }
});

document.getElementById('btn-sound').addEventListener('click', () => {
  soundEnabled = !soundEnabled;
  document.getElementById('btn-sound').innerText = `🔊 เสียง: ${soundEnabled ? 'เปิด' : 'ปิด'}`;
});

document.getElementById('btn-side-white').addEventListener('click', () => {
  humanSide = 1;
  document.getElementById('btn-side-white').className = 'py-2 px-3 rounded-xl bg-amber-600 font-semibold text-xs transition border border-amber-500';
  document.getElementById('btn-side-black').className = 'py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs transition border border-slate-700';
  resetGame();
});

document.getElementById('btn-side-black').addEventListener('click', () => {
  humanSide = -1;
  document.getElementById('btn-side-black').className = 'py-2 px-3 rounded-xl bg-amber-600 font-semibold text-xs transition border border-amber-500';
  document.getElementById('btn-side-white').className = 'py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs transition border border-slate-700';
  resetGame();
});

// App Entry Point
createBoardDOM();
fetchState();

// ========================================================
// Live AI Training Dashboard & Real-Time Telemetry
// ========================================================

const tabPlay = document.getElementById('tab-play-mode');
const tabTrain = document.getElementById('tab-train-mode');
const panelPlay = document.getElementById('panel-play-mode');
const panelTrain = document.getElementById('panel-train-mode');

tabPlay.addEventListener('click', () => {
  tabPlay.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-amber-600 text-white transition flex items-center gap-1.5 shadow';
  tabTrain.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200 transition flex items-center gap-1.5';
  panelPlay.className = 'block w-full';
  panelTrain.className = 'hidden w-full';
});

tabTrain.addEventListener('click', () => {
  tabTrain.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-amber-600 text-white transition flex items-center gap-1.5 shadow';
  tabPlay.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200 transition flex items-center gap-1.5';
  panelTrain.className = 'block w-full';
  panelPlay.className = 'hidden w-full';
  initTrainingDashboard();
});

// Chart.js instance for Loss
let lossChart = null;

function initLossChart() {
  if (lossChart) return;
  const ctx = document.getElementById('lossChart');
  if (!ctx) return;
  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Total Loss',
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 3,
        },
        {
          label: 'Policy Loss',
          data: [],
          borderColor: '#06b6d4',
          backgroundColor: 'transparent',
          tension: 0.3,
          borderWidth: 1.5,
          pointRadius: 2,
        },
        {
          label: 'Value Loss',
          data: [],
          borderColor: '#10b981',
          backgroundColor: 'transparent',
          tension: 0.3,
          borderWidth: 1.5,
          pointRadius: 2,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { family: 'Prompt, sans-serif', size: 10 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { family: 'Prompt, sans-serif', size: 10 } }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#cbd5e1', font: { family: 'Prompt, sans-serif', size: 11 } }
        }
      }
    }
  });
}

function updateLossChart(history) {
  if (!lossChart) initLossChart();
  if (!lossChart || !history || !history.loss) return;
  const n = history.loss.length;
  lossChart.data.labels = Array.from({ length: n }, (_, i) => `Iter ${i + 1}`);
  lossChart.data.datasets[0].data = history.loss;
  lossChart.data.datasets[1].data = history.policy_loss;
  lossChart.data.datasets[2].data = history.value_loss;
  lossChart.update();
}

function createTrainBoardDOM() {
  const boardEl = document.getElementById('train-board');
  if (!boardEl || boardEl.children.length > 0) return;
  boardEl.innerHTML = '';
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = document.createElement('div');
      const isPlayable = (r + c) % 2 === 0;
      cell.className = `board-cell ${isPlayable ? 'bg-board-dark' : 'bg-board-light'}`;
      cell.id = `train-cell-${r}-${c}`;
      boardEl.appendChild(cell);
    }
  }
}

function renderTrainBoard(squares, fromSq, toSq) {
  if (!squares) return;
  createTrainBoardDOM();

  document.querySelectorAll('#train-board .board-cell').forEach(cell => {
    cell.innerHTML = '';
    cell.classList.remove('selected-sq');
  });

  for (let sq = 0; sq < 32; sq++) {
    const pieceVal = squares[sq];
    if (pieceVal !== 0) {
      const [r, c] = SQ_TO_COORD[sq];
      const cell = document.getElementById(`train-cell-${r}-${c}`);
      if (cell) {
        const pEl = document.createElement('div');
        const isWhite = pieceVal > 0;
        const isKing = Math.abs(pieceVal) === 2;
        pEl.className = `piece ${isWhite ? 'piece-p1' : 'piece-p2'}`;
        if (isKing) {
          pEl.innerHTML = '<span class="king-badge">👑</span>';
        }
        cell.appendChild(pEl);
      }
    }
  }

  // Highlight last move
  if (fromSq !== undefined && toSq !== undefined && SQ_TO_COORD[fromSq] && SQ_TO_COORD[toSq]) {
    const [r1, c1] = SQ_TO_COORD[fromSq];
    const [r2, c2] = SQ_TO_COORD[toSq];
    const cFrom = document.getElementById(`train-cell-${r1}-${c1}`);
    const cTo = document.getElementById(`train-cell-${r2}-${c2}`);
    if (cFrom) cFrom.classList.add('selected-sq');
    if (cTo) cTo.classList.add('selected-sq');
  }
}

// WebSocket connection for real-time training events
let trainSocket = null;
let currentVisualDelay = 0.0; // Default: Turbo

function connectTrainingWS() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/training`;

  trainSocket = new WebSocket(wsUrl);

  trainSocket.onopen = () => {
    appendTrainLog('🔌 เชื่อมต่อ WebSocket ฝึกฝน AI สำเร็จ');
  };

  trainSocket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleTrainMessage(msg);
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };

  trainSocket.onclose = () => {
    setTimeout(connectTrainingWS, 2500);
  };
}

function updateSelfPlayStats(stats) {
  if (!stats) return;
  const whiteWins = stats.white_wins || 0;
  const blackWins = stats.black_wins || 0;
  const draws = stats.draws || 0;
  const total = stats.total_games || (whiteWins + blackWins + draws);

  const whitePct = total > 0 ? ((whiteWins / total) * 100).toFixed(1) : (0.0).toFixed(1);
  const blackPct = total > 0 ? ((blackWins / total) * 100).toFixed(1) : (0.0).toFixed(1);
  const drawPct = total > 0 ? ((draws / total) * 100).toFixed(1) : (0.0).toFixed(1);

  const elWhite = document.getElementById('stat-white-wins');
  const elBlack = document.getElementById('stat-black-wins');
  const elDraws = document.getElementById('stat-draws');
  const elTotal = document.getElementById('stat-total-games');
  const barWhite = document.getElementById('bar-stat-white');
  const barBlack = document.getElementById('bar-stat-black');
  const barDraw = document.getElementById('bar-stat-draw');

  if (elWhite) elWhite.innerText = `${whiteWins} (${whitePct}%)`;
  if (elBlack) elBlack.innerText = `${blackWins} (${blackPct}%)`;
  if (elDraws) elDraws.innerText = `${draws} (${drawPct}%)`;
  if (elTotal) elTotal.innerText = `${total} เกม`;

  if (barWhite) barWhite.style.width = `${whitePct}%`;
  if (barBlack) barBlack.style.width = `${blackPct}%`;
  if (barDraw) barDraw.style.width = `${drawPct}%`;
}

function handleTrainMessage(msg) {
  if (msg.type === 'init') {
    updateTrainUIState(msg.is_training);
    document.getElementById('train-iter-display').innerText = `${msg.current_iter} / ${msg.total_iters}`;
    document.getElementById('train-ep-display').innerText = `${msg.current_episode} / ${msg.total_episodes}`;
    document.getElementById('train-buffer-display').innerText = `${msg.buffer_size} Samples`;
    if (msg.stats) {
      updateSelfPlayStats(msg.stats);
    }
    if (msg.history && msg.history.loss && msg.history.loss.length > 0) {
      updateLossChart(msg.history);
      const lastWr = msg.history.win_rate[msg.history.win_rate.length - 1];
      if (lastWr !== undefined) {
        document.getElementById('train-winrate-display').innerText = `${(lastWr * 100).toFixed(1)}%`;
      }
    }
  } else if (msg.type === 'step') {
    const d = msg.data;
    document.getElementById('train-iter-display').innerText = `${d.iteration} / ${d.total_iters}`;
    document.getElementById('train-ep-display').innerText = `${d.episode} / ${d.total_episodes}`;
    document.getElementById('train-buffer-display').innerText = `${d.buffer_size} Samples`;
    document.getElementById('train-last-move').innerText = `${d.acting_player === 1 ? '⚪ ขาว' : '⚫ ดำ'}: ${d.notation}`;
    document.getElementById('train-turn-dot').className = `w-3 h-3 rounded-full ${d.acting_player === 1 ? 'bg-amber-400' : 'bg-slate-400'} animate-pulse`;
    renderTrainBoard(d.squares, d.from_sq, d.to_sq);
    playSound('move');
  } else if (msg.type === 'episode_end') {
    document.getElementById('train-ep-display').innerText = `${msg.episode} / ${msg.total_episodes}`;
    document.getElementById('train-buffer-display').innerText = `${msg.buffer_size} Samples`;
    if (msg.stats) {
      updateSelfPlayStats(msg.stats);
    }
    const outcomeStr = msg.winner_label ? `ผลลัพธ์: ${msg.winner_label}` : '';
    const reasonStr = msg.reason ? `(${msg.reason})` : '';
    const badgeEl = document.getElementById('train-last-move');
    if (badgeEl) {
      badgeEl.innerText = `🏁 ${msg.winner_label || 'จบเกม'} ${reasonStr}`;
      badgeEl.className = 'text-xs font-mono font-semibold px-2.5 py-1 rounded bg-amber-950 text-amber-300 border border-amber-700 animate-pulse';
    }
    appendTrainLog(`🎮 จบเกมจำลองที่ ${msg.episode}/${msg.total_episodes}: ${outcomeStr} ${reasonStr} (สะสมในสมองแล้ว: ${msg.buffer_size} ตำแหน่ง)`);
  } else if (msg.type === 'epoch_end') {
    appendTrainLog(`🧠 Epoch ${msg.epoch}: Total Loss = ${msg.loss.toFixed(4)} (Policy = ${msg.policy_loss.toFixed(4)}, Value = ${msg.value_loss.toFixed(4)})`);
  } else if (msg.type === 'iter_end') {
    updateLossChart(msg.history);
    const wr = msg.info.win_rate * 100;
    document.getElementById('train-winrate-display').innerText = `${wr.toFixed(1)}%`;
    appendTrainLog(`⚔️ สิ้นสุดรอบที่ ${msg.iteration}! ผลดวล Arena: ชนะ ${wr.toFixed(1)}% ${msg.info.accepted ? '🏆 (ผ่านเกณฑ์ - บันทึกเป็นโมเดลตัวเก่งที่สุด)' : '🛡️ (ใช้โมเดลเดิม)'}`);
  } else if (msg.type === 'training_complete') {
    updateTrainUIState(false);
    appendTrainLog('🎉 การฝึกฝนเสร็จสิ้นเรียบร้อยแล้ว! สมองเวอร์ชันใหม่พร้อมใช้งานในโหมดเล่นเกม');
    updateLossChart(msg.history);
  } else if (msg.type === 'log') {
    appendTrainLog(msg.message);
  }
}

function updateTrainUIState(isTraining) {
  const btnStart = document.getElementById('btn-start-training');
  const btnStop = document.getElementById('btn-stop-training');
  const statusText = document.getElementById('train-status-text');

  if (isTraining) {
    btnStart.disabled = true;
    btnStart.className = 'py-3 px-4 rounded-xl bg-slate-800 text-slate-500 font-bold text-sm transition border border-slate-700 flex items-center justify-center gap-2 cursor-not-allowed';
    btnStop.disabled = false;
    btnStop.className = 'py-3 px-4 rounded-xl bg-red-600 hover:bg-red-500 font-bold text-sm text-white transition shadow-lg shadow-red-950/50 flex items-center justify-center gap-2 cursor-pointer';
    statusText.innerText = '⚡ กำลังฝึกฝน AI (Training...)';
    statusText.className = 'text-sm font-bold text-emerald-400 mt-1 animate-pulse';
  } else {
    btnStart.disabled = false;
    btnStart.className = 'py-3 px-4 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 font-bold text-sm text-white transition shadow-lg shadow-emerald-950/50 flex items-center justify-center gap-2 cursor-pointer';
    btnStop.disabled = true;
    btnStop.className = 'py-3 px-4 rounded-xl bg-slate-800 text-slate-500 font-bold text-sm transition border border-slate-700 flex items-center justify-center gap-2 cursor-not-allowed';
    statusText.innerText = 'พร้อมเริ่มการฝึกฝน (Idle)';
    statusText.className = 'text-sm font-bold text-amber-400 mt-1';
  }
}

function appendTrainLog(text) {
  const logEl = document.getElementById('train-logs');
  if (!logEl) return;
  const now = new Date().toLocaleTimeString();
  const p = document.createElement('div');
  p.innerHTML = `<span class="text-slate-500">[${now}]</span> ${text}`;
  logEl.appendChild(p);
  logEl.scrollTop = logEl.scrollHeight;
}

function initTrainingDashboard() {
  createTrainBoardDOM();
  initLossChart();
  if (!trainSocket || trainSocket.readyState !== WebSocket.OPEN) {
    connectTrainingWS();
  }
}

// Speed Mode Buttons
const btnTurbo = document.getElementById('btn-speed-turbo');
const btnVisual = document.getElementById('btn-speed-visual');

btnTurbo.addEventListener('click', () => {
  currentVisualDelay = 0.0;
  btnTurbo.className = 'py-2 px-3 rounded-xl bg-amber-600 font-semibold text-xs transition border border-amber-500 text-white';
  btnVisual.className = 'py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs transition border border-slate-700';
});

btnVisual.addEventListener('click', () => {
  currentVisualDelay = 0.08;
  btnVisual.className = 'py-2 px-3 rounded-xl bg-amber-600 font-semibold text-xs transition border border-amber-500 text-white';
  btnTurbo.className = 'py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs transition border border-slate-700';
});

// Training Start / Stop Buttons
document.getElementById('btn-start-training').addEventListener('click', async () => {
  const iters = parseInt(document.getElementById('input-train-iters').value) || 5;
  const eps = parseInt(document.getElementById('input-train-episodes').value) || 5;
  const sims = parseInt(document.getElementById('input-train-sims').value) || 50;
  const resume = document.getElementById('chk-resume').checked;

  try {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        num_iters: iters,
        episodes_per_iter: eps,
        mcts_sims: sims,
        visual_delay: currentVisualDelay,
        resume: resume,
      })
    });
    if (res.ok) {
      updateTrainUIState(true);
    } else {
      const err = await res.json();
      alert(`ไม่สามารถเริ่มการเทรนได้: ${err.detail || 'เกิดข้อผิดพลาด'}`);
    }
  } catch (e) {
    console.error(e);
    alert('เกิดข้อผิดพลาดในการเชื่อมต่อเซิร์ฟเวอร์');
  }
});

document.getElementById('btn-stop-training').addEventListener('click', async () => {
  try {
    const res = await fetch('/api/train/stop', { method: 'POST' });
    if (res.ok) {
      appendTrainLog('🛑 กำลังส่งคำสั่งหยุดการเทรน...');
    }
  } catch (e) {
    console.error(e);
  }
});

document.getElementById('btn-clear-logs').addEventListener('click', () => {
  const logEl = document.getElementById('train-logs');
  if (logEl) logEl.innerHTML = '';
});

// Connect WebSocket on boot in background
connectTrainingWS();
