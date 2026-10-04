# -*- coding: utf-8 -*-
"""
Ймовірнісно-вартісна модель стратегій перевірки фінальності для легких клієнтів
PoS-блокчейнів (S1 повна перевірка, S2 рекурсивний доказ, S3 випадкова
вибірка k підписів, S4 штатний легкий клієнт NEAR).

Один скрипт генерує всі таблиці (results/*.csv) і рисунки (figures/*.png).
Запуск:  python finality_model.py          (робочий каталог = каталог скрипта)
Зерно генератора фіксоване (SEED). Нових запусків Plonky2/Rust немає:
виміри S2 узято з опублікованих праць як вхідні параметри (див. PARAMS).
"""
import json, math, os, sys, time
import numpy as np
import pandas as pd
from scipy.special import gammaln

SEED = 20261004
RNG = np.random.default_rng(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Параметри. Кожен має джерело; "ASSUMPTION" = припущення, досліджене діапазоном.
# ----------------------------------------------------------------------------
PARAMS = [
 # name, value, unit, source, kind
 ("proof_bytes_final", 180112, "B", "Kuznetsov et al., 2024b (CEUR-WS 3826), p.101-102", "published"),
 ("proof_bytes_sig", 133080, "B", "Kuznetsov et al., 2024b (CEUR-WS 3826), p.100", "published"),
 ("t_verify_S2_ms_lo", 4.0, "ms", "CEUR-WS 3826, p.100 (block-hash proof, ~4 ms)", "published"),
 ("t_verify_S2_ms_nom", 4.5, "ms", "CEUR-WS 3826, p.101 (next_bp_hash proof, ~4.5 ms)", "published"),
 ("t_verify_S2_ms_hi", 5.1, "ms", "CEUR-WS 3826, p.100 (signature proof, 4.6-5.1 ms)", "published"),
 ("t_final_aggregation_s", 0.47, "s", "CEUR-WS 3826, p.101 (final aggregation step = part of proof generation; p.102 calls it light-client verification) - NOT USED", "published"),
 ("t_verify_rec_ryzen7_ms", "5.5-7.0", "ms", "Kuznetsov et al., 2024a (LNNS 1091), p.275, Table 4", "published"),
 ("clk_GHz_lo", 4.5, "GHz", "CEUR-WS 3826, p.99 (Ryzen 9 7950X base)", "published"),
 ("clk_GHz_nom", 4.7, "GHz", "CEUR-WS 3826, p.99 (stated clock)", "published"),
 ("clk_GHz_hi", 5.7, "GHz", "CEUR-WS 3826, p.99 (max boost)", "published"),
 ("t_gen_S2_s", "806-1073", "s", "CEUR-WS 3826, p.101 (prover side; not part of per-act verifier cost)", "published"),
 ("sha256_cycles_per_byte", 15, "cycles/B", "LNNS 1091, p.275 (Crypto++ benchmark cited there)", "published"),
 ("ed_cycles_single", 273364, "cycles", "Bernstein et al., 2012, p.78 (Westmere)", "external-verified"),
 ("ed_cycles_batch", 134000, "cycles", "Bernstein et al., 2012, p.78 (batch of 64)", "external-verified"),
 ("lambda_ed25519", 128, "bit", "Bernstein et al., 2012 (2^128 security target)", "external-verified"),
 ("lambda_S2", 100, "bit", "Plonky2 README, Security: default FRI = 100 bits conjectured (ethSTARK conjecture)", "external-verified"),
 ("lambda_poseidon_plonky2", 95, "bit", "Plonky2 README citing Bariant et al., 2022", "external-verified"),
 ("sig_bytes", 64, "B", "RFC 8032 (Ed25519 signature)", "external-verified"),
 ("pk_bytes", 32, "B", "RFC 8032 (Ed25519 public key)", "external-verified"),
 ("stake_bytes", 16, "B", "nearcore views.rs: stake u128", "external-verified"),
 ("near_epoch_length", 43200, "blocks", "NEAR RPC EXPERIMENTAL_protocol_config, 2026-10-04", "external-verified"),
 ("near_bp_seats", 100, "validators", "NEAR RPC EXPERIMENTAL_protocol_config, 2026-10-04", "external-verified"),
 ("near_lcb_json_bytes", None, "B", "NEAR RPC next_light_client_block, 2026-10-04 (measured)", "external-measured"),
 ("TX_BASE", 21000, "gas", "ethereum/execution-specs prague gas.py; EIP-7623", "external-verified"),
 ("TOKEN_STD", 4, "gas/token", "EIP-7623 STANDARD_TOKEN_COST (=16 gas/non-zero byte, EIP-2028)", "external-verified"),
 ("TOKEN_FLOOR", 10, "gas/token", "EIP-7623 TOTAL_COST_FLOOR_PER_TOKEN (Prague, EIP-7600)", "external-verified"),
 ("ECADD", 150, "gas", "EIP-1108", "external-verified"),
 ("ECMUL", 6000, "gas", "EIP-1108", "external-verified"),
 ("PAIR_BASE", 45000, "gas", "EIP-1108", "external-verified"),
 ("PAIR_PER_POINT", 34000, "gas", "EIP-1108", "external-verified"),
 ("KECCAK_BASE", 30, "gas", "execution-specs gas.py", "external-verified"),
 ("KECCAK_WORD", 6, "gas", "execution-specs gas.py", "external-verified"),
 ("SHA256_BASE", 60, "gas", "execution-specs gas.py", "external-verified"),
 ("SHA256_WORD", 12, "gas", "execution-specs gas.py", "external-verified"),
 ("STORAGE_SET", 20000, "gas", "execution-specs gas.py", "external-verified"),
 ("ECRECOVER", 3000, "gas", "execution-specs gas.py", "external-verified"),
 ("G_ed_EIP665", 2000, "gas", "EIP-665 proposed ED25519VFY price (status Stagnant, not deployed)", "external-verified"),
 ("G_S2_ultraplonk_hi", 943720, "gas", "Bhatt et al., 2025, Fig.3 (UltraPLONK, |V|=1e6; ePrint 2025/057 p.16)", "external-verified"),
 ("G_S2_ultraplonk_lo", 797216, "gas", "Bhatt et al., 2025, Fig.3 (UltraPLONK, |V|=10)", "external-verified"),
 ("groth16_public_inputs", 3, "count", "ASSUMPTION A3: Plonky2 proof wrapped into Groth16/BN254 with 3 public inputs", "assumption"),
 ("q_participation", 0.71, "fraction", "ASSUMPTION A2 (= share of non-empty approvals in RPC snapshot; range 0.67-1.0)", "assumption"),
 ("account_id_bytes_generic", 20, "B", "ASSUMPTION A8 (generic chains); NEAR uses real ids from snapshot", "assumption"),
 ("grinding_r", "0; 2^10; 2^64", "realizations", "ASSUMPTION A5 (local / on-chain RANDAO after Bhatt et al. / non-interactive Fiat-Shamir)", "assumption"),
 ("G_ed_pure_EVM", "1e5-1e6", "gas", "ASSUMPTION A4 (no primary source; EIP-665 only says 'very high')", "assumption"),
]
P = {r[0]: r[1] for r in PARAMS}

LAMBDA_ED = 128.0
LAMBDA_S2 = 100.0
EPS_BITS = [40, 64, 80, 100, 128]
R_SET = {"r = 0 (локальна випадковість)": 0.0, "r = 2^10 (RANDAO)": 2.0**10, "r = 2^64 (Фіат–Шамір)": 2.0**64}

# ----------------------------------------------------------------------------
# 2. Знімок NEAR (RPC, 2026-10-04): стейки 100 виробників блоків, учасники підпису
# ----------------------------------------------------------------------------
def load_near():
    prev = json.load(open(os.path.join(DATA, "near_lcb_prev.json")))["result"]
    cur = json.load(open(os.path.join(DATA, "near_lcb.json")))["result"]
    bps = prev["next_bps"]            # набір виробників блоків поточної епохи
    stakes = np.array([int(b["stake"]) for b in bps], dtype=float)
    ids = [b["account_id"] for b in bps]
    present = np.array([a is not None for a in cur["approvals_after_next"]])
    next_ids = [b["account_id"] for b in cur["next_bps"]]
    json_bytes = os.path.getsize(os.path.join(DATA, "near_lcb.json"))
    return dict(stakes=stakes / stakes.sum(), ids=ids, present=present,
                next_ids=next_ids, json_bytes=json_bytes,
                height=cur["inner_lite"]["height"], epoch_id=cur["inner_lite"]["epoch_id"])

NEAR = load_near()
P["near_lcb_json_bytes"] = NEAR["json_bytes"]

# ----------------------------------------------------------------------------
# 3. Модель безпеки
# ----------------------------------------------------------------------------
def log2_comb(a, b):
    return (gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)) / math.log(2)

def p_hyper(n, f, k):
    """Рівні стейки, вибірка без повернення з заявленої множини.
    Заявлена множина: m = floor(2n/3)+1 (поріг > 2/3), з них a = floor(f n) зловмисних.
    Повертає log2 P (−inf, якщо P = 0)."""
    m = n * 2 // 3 + 1
    a = int(math.floor(f * n + 1e-9))
    if a >= m:
        return 0.0
    if k > a:
        return -np.inf
    return log2_comb(a, k) - log2_comb(m, k)

def p_binom_bound(f, k):
    """Межа з поверненням (n→∞): P = (3f/2)^k (заявлено рівно 2/3)."""
    p = min(1.0, 1.5 * f)
    return k * math.log2(p) if p > 0 else -np.inf

def grind_bits(log2p, r):
    """−log2 P_r,  P_r = 1 − (1 − P)^(r+1) (адверсарій відкидає r реалізацій)."""
    if log2p == -np.inf:
        return np.inf
    P1 = 2.0 ** log2p
    if P1 == 0.0:  # підтікання: використовуємо (r+1)P
        return -(log2p + math.log2(r + 1))
    Pr = -math.expm1((r + 1) * math.log1p(-min(P1, 1 - 1e-300))) if P1 < 1 else 1.0
    if Pr <= 0:
        return -(log2p + math.log2(r + 1))
    return -math.log2(min(1.0, Pr))

from functools import lru_cache

@lru_cache(maxsize=None)
def kstar_equal(n, f, eps_bits, r=0.0, kmax=None):
    m = n * 2 // 3 + 1
    a = int(math.floor(f * n + 1e-9))
    if a >= m:
        return np.nan
    kmax = a + 1 if kmax is None else kmax
    for k in range(1, kmax + 1):
        if grind_bits(p_hyper(n, f, k), r) >= eps_bits:
            return k
    return a + 1

def kstar_binom(f, eps_bits, r=0.0):
    p = 1.5 * f
    if p >= 1:
        return np.nan
    return int(math.ceil((eps_bits + math.log2(r + 1)) / (-math.log2(p))))

# --- стейк-зважені набори ----------------------------------------------------
def zipf_stakes(n, alpha):
    s = np.arange(1, n + 1, dtype=float) ** (-alpha)
    return s / s.sum()

def pareto_stakes(n, shape):
    # детерміновані квантилі розподілу Парето (x_m = 1): x_i = (1 - (i-0.5)/n)^(-1/shape)
    u = (np.arange(1, n + 1) - 0.5) / n
    s = np.sort((1 - u) ** (-1.0 / shape))[::-1]
    return s / s.sum()

def corrupt(stakes, f, strategy, rng=None):
    """Адверсарій отримує контроль над валідаторами із сумарним стейком ≤ f."""
    n = len(stakes)
    if strategy == "top":
        order = np.argsort(-stakes, kind="stable")
    elif strategy == "bottom":
        order = np.argsort(stakes, kind="stable")
    else:
        order = rng.permutation(n)
    adv = np.zeros(n, bool); tot = 0.0
    for i in order:
        if tot + stakes[i] <= f + 1e-12:
            adv[i] = True; tot += stakes[i]
    return adv

def claim_honest(stakes, adv, mode="stake"):
    """Чесні валідатори, яких адверсарій «дописує» до заявленої множини, щоб
    перевищити поріг 2/3 стейку. mode='stake' — мінімізує сумарний чесний стейк H
    (оптимально для зваженої вибірки); mode='count' — мінімізує кількість (для
    вибірки без урахування стейку)."""
    A = stakes[adv].sum()
    need = 2.0 / 3.0 - A
    hon = np.where(~adv)[0]
    if need < 0:
        return np.zeros(len(stakes), bool)
    hs = hon[np.argsort(-stakes[hon], kind="stable")]
    claimed = np.zeros(len(stakes), bool); tot = 0.0
    for i in hs:
        if tot > need + 1e-12:
            break
        claimed[i] = True; tot += stakes[i]
    if mode == "stake":
        # покращення: замінити останнього доданого найменшим, що все ще перевищує поріг
        last = [i for i in hs if claimed[i]][-1]
        rest = tot - stakes[last]
        cand = [i for i in hon if (not claimed[i]) or i == last]
        cand = [i for i in cand if rest + stakes[i] > need + 1e-12]
        if cand:
            best = min(cand, key=lambda i: stakes[i])
            claimed[last] = False; claimed[best] = True
    return claimed

def p_weighted_noreplace(adv_st, H, K, n_t=3000):
    """Точна P(k перших зважених вибірок без повернення — усі зловмисні), k=1..K.
    Подання через експоненційні «годинники» (еквівалент Ефраїмідіса–Спіракіса):
    P(k) = ∫ H e^{−Ht} Pr[N_A(t) ≥ k] dt,  N_A(t) — пуассон-біноміальна з p_i=1−e^{−s_i t}.
    Повертає масив log2 P(k)."""
    adv_st = np.asarray(adv_st, float)
    nA = len(adv_st)
    out = np.full(K, -np.inf)
    if nA == 0 or H <= 0:
        if H <= 0:
            out[:] = 0.0
        return out
    Kc = min(K, nA)
    smin = adv_st.min()
    t = np.geomspace(1e-6 / adv_st.max(), 1500.0 / H, n_t)
    lt = np.log(t)
    dist = np.zeros((n_t, Kc + 1)); dist[:, 0] = 1.0
    for s in adv_st:
        p = -np.expm1(-s * t)[:, None]
        new = dist * (1 - p)
        new[:, 1:] += dist[:, :-1] * p
        new[:, Kc] += dist[:, Kc] * p[:, 0]  # поглинальний стан "≥ Kc"
        dist = new
    tail = np.cumsum(dist[:, ::-1], axis=1)[:, ::-1]  # tail[:, j] = Pr[N ≥ j]
    w = H * np.exp(-H * t) * t  # заміна змінної: dt = t dln t
    for k in range(1, Kc + 1):
        y = w * tail[:, k]
        val = np.trapezoid(y, lt) if hasattr(np, "trapezoid") else np.trapz(y, lt)
        out[k - 1] = math.log2(val) if val > 0 else -np.inf
    return out

def mc_weighted_noreplace(stakes, claimed_mask, adv_mask, k, trials, rng):
    """Монте-Карло: зважена вибірка без повернення через ключі −ln(U)/w."""
    idx = np.where(claimed_mask)[0]
    w = stakes[idx]; isadv = adv_mask[idx]
    hits = 0; B = 20000
    done = 0
    while done < trials:
        b = min(B, trials - done)
        keys = -np.log(rng.random((b, len(idx)))) / w
        first = np.argpartition(keys, k - 1, axis=1)[:, :k] if k < len(idx) else np.tile(np.arange(len(idx)), (b, 1))
        hits += int(np.all(isadv[first], axis=1).sum())
        done += b
    return hits / trials

def weighted_case(stakes, f, strategy, K, rng=None):
    adv = corrupt(stakes, f, strategy, rng)
    A = stakes[adv].sum()
    claimed_h = claim_honest(stakes, adv, "stake")
    H = stakes[claimed_h].sum()
    lp_w = p_weighted_noreplace(stakes[adv], H, K)
    # з поверненням
    pr = A / (A + H)
    lp_wr = np.array([k * math.log2(pr) for k in range(1, K + 1)])
    # без урахування стейку (рівноймовірно серед заявлених), чесні — мінімальна кількість
    claimed_c = claim_honest(stakes, adv, "count")
    a_c, m_c = int(adv.sum()), int(adv.sum() + claimed_c.sum())
    Ku = max(K, a_c + 1)
    lp_u = np.array([(log2_comb(a_c, k) - log2_comb(m_c, k)) if k <= a_c else -np.inf for k in range(1, Ku + 1)])
    return dict(adv=adv, claimed=claimed_h | adv, A=A, H=H, n_adv=int(adv.sum()),
                lp_w=lp_w, lp_wr=lp_wr, lp_u=lp_u, m_c=m_c)

def kstar_from_curve(lp, eps_bits, r=0.0):
    for k, l in enumerate(lp, start=1):
        if grind_bits(l, r) >= eps_bits:
            return k
    return np.nan

# ----------------------------------------------------------------------------
# 4. Модель вартості одного кроку синхронізації
#    (прийняти фіналізований заголовок нової епохи + перехід набору валідаторів)
# ----------------------------------------------------------------------------
HDR = 32 + 32 + 208 + 32   # LightClientBlockView без списків (nomicon; views.rs)

def ed_cycles(batch=False):
    return P["ed_cycles_batch"] if batch else P["ed_cycles_single"]

def quorum(n):
    return n * 2 // 3 + 1

def n_min_quorum(stakes, present=None):
    """S1′: кількість підписів, які треба перевірити, щоб перевищити 2/3 стейку,
    якщо перевіряти наявні підписи в порядку спадання стейку (безпека як у S1)."""
    st = np.asarray(stakes, float)
    if present is not None:
        st = st[np.asarray(present, bool)]
    cs = np.cumsum(np.sort(st)[::-1])
    return int(np.searchsorted(cs, 2.0 / 3.0 * np.sum(stakes), side="right") + 1)

def bytes_valset(n, id_bytes=None, ids=None):
    """Borsh(Vec<ValidatorStakeView>): 4 + Σ(1 тег + 4 + |id| + 1+32 ключ + 16 стейк)."""
    if ids is not None:
        return 4 + sum(1 + 4 + len(x.encode()) + 33 + 16 for x in ids)
    idb = P["account_id_bytes_generic"] if id_bytes is None else id_bytes
    return 4 + n * (1 + 4 + idb + 33 + 16)

def cost_S1(n, batch=False):
    m = quorum(n)
    vs = bytes_valset(n)
    by = HDR + 4 + math.ceil(n / 8) + m * 64 + vs
    cyc = m * ed_cycles(batch) + vs * P["sha256_cycles_per_byte"]
    return dict(sigs=m, bytes=by, cycles=cyc)

def cost_S4(n, q=None, ids=None, present=None, batch=False):
    q = P["q_participation"] if q is None else q
    npres = int(present.sum()) if present is not None else int(math.ceil(q * n))
    vs = bytes_valset(n, ids=ids)
    by = HDR + 1 + vs + 4 + n * 1 + npres * 65
    cyc = npres * ed_cycles(batch) + vs * P["sha256_cycles_per_byte"]
    return dict(sigs=npres, bytes=by, cycles=cyc)

def cost_S2(variant="nom"):
    t = {"lo": P["t_verify_S2_ms_lo"], "nom": P["t_verify_S2_ms_nom"], "hi": P["t_verify_S2_ms_hi"]}[variant] * 1e-3
    clk = P["clk_GHz_nom"] * 1e9            # одна робоча частота стенда, 4,7 ГГц (CEUR-WS 3826, p.99)
    pb = P["proof_bytes_final"]              # S2 засвідчує і перехід набору валідаторів -> агрегований доказ 180 112 Б
    return dict(sigs=0, bytes=pb + 72, cycles=t * clk)

def cost_S3(n, k, merkle=True, ids=None):
    """S3 за Bhatt et al.: бітова карта + k підписів з відкриттями зобов'язання.
    merkle=True — набір валідаторів зобов'язано Меркл-коренем (як у Bhatt et al.);
    merkle=False — поточний NEAR: next_bp_hash = SHA-256(borsh(next_bps)) — плоский
    хеш, тож клієнт мусить завантажити повний next_bps."""
    depth = max(1, math.ceil(math.log2(n)))
    if merkle:
        per = 64 + 33 + 16 + 32 * depth
        by = HDR + math.ceil(n / 8) + k * per + 32
        cyc = k * ed_cycles() + k * depth * 64 * P["sha256_cycles_per_byte"]
    else:
        vs = bytes_valset(n, ids=ids)
        by = HDR + 1 + vs + math.ceil(n / 8) + k * 65
        cyc = k * ed_cycles() + vs * P["sha256_cycles_per_byte"]
    return dict(sigs=k, bytes=by, cycles=cyc)

# --- газ EVM ----------------------------------------------------------------
def tx_gas(calldata_bytes, exec_gas, nonzero_frac=1.0):
    nz = calldata_bytes * nonzero_frac; z = calldata_bytes - nz
    tokens = z + 4 * nz
    return P["TX_BASE"] + max(P["TOKEN_STD"] * tokens + exec_gas, P["TOKEN_FLOOR"] * tokens)

def sha256_gas(nbytes):
    return P["SHA256_BASE"] + P["SHA256_WORD"] * math.ceil(nbytes / 32)

def keccak_gas(nbytes):
    return P["KECCAK_BASE"] + P["KECCAK_WORD"] * math.ceil(nbytes / 32)

def gas_S1(n, G_ed):
    m = quorum(n); vs = n * 48
    cd = HDR + math.ceil(n / 8) + m * 64 + 2 * vs   # поточний і наступний набори в calldata
    ex = m * G_ed + 2 * sha256_gas(vs) + P["STORAGE_SET"]   # + запис зобов'язання нового набору
    return tx_gas(cd, ex)

def gas_S3(n, k, G_ed):
    depth = max(1, math.ceil(math.log2(n)))
    per = 64 + 48 + 32 * depth
    commit = tx_gas(HDR + math.ceil(n / 8) + per, G_ed + depth * keccak_gas(64) + keccak_gas(48) + 2 * P["STORAGE_SET"])
    reveal = tx_gas(k * per, k * (G_ed + depth * keccak_gas(64) + keccak_gas(48)) + P["STORAGE_SET"])  # + запис нового набору
    return commit + reveal

def gas_S2_groth16(l=None):
    l = P["groth16_public_inputs"] if l is None else l
    ex = P["PAIR_BASE"] + 4 * P["PAIR_PER_POINT"] + l * (P["ECMUL"] + P["ECADD"]) + P["STORAGE_SET"]  # + запис нового набору
    return tx_gas(256 + 32 * l, ex)

def gas_S2_direct_lower(pb=None):
    pb = P["proof_bytes_final"] if pb is None else pb
    return P["TX_BASE"] + P["TOKEN_FLOOR"] * 4 * pb   # лише calldata, нижня межа

GAS_S2 = {"Groth16 (A3)": gas_S2_groth16(), "UltraPLONK (Bhatt et al.)": P["G_S2_ultraplonk_hi"] + P["STORAGE_SET"]}

# ----------------------------------------------------------------------------
# 5. Обчислення
# ----------------------------------------------------------------------------
def fmt_bits(x):
    return "∞ (детерм.)" if not np.isfinite(x) else f"{x:.1f}"

def run_tables():
    out = {}
    pd.DataFrame(PARAMS, columns=["parameter", "value", "unit", "source", "kind"]).assign(
        value=lambda d: d["value"].astype(str)).to_csv(os.path.join(RES, "T0_parameters.csv"), index=False)

    # --- T1: k*(ε) для рівних стейків ---
    rows = []
    for n in [100, 1000, 10000, None]:
        for f in [0.20, 0.25, 0.30, 1 / 3]:
            for rname, r in R_SET.items():
                row = dict(n="∞ (з поверненням)" if n is None else n, f=round(f, 4), grinding=rname)
                for e in EPS_BITS:
                    row[f"k*(2^-{e})"] = kstar_binom(f, e, r) if n is None else kstar_equal(n, f, e, r)
                if n is not None:
                    row["k_det = a+1"] = int(math.floor(f * n + 1e-9)) + 1
                    row["quorum ⌊2n/3⌋+1"] = quorum(n)
                rows.append(row)
    T1 = pd.DataFrame(rows); T1.to_csv(os.path.join(RES, "T1_kstar_equal_stake.csv"), index=False)
    out["T1"] = T1

    # --- T2: стейк-зважені розподіли ---
    dists = {
        "рівні (n=100)": np.full(100, 0.01),
        "NEAR, знімок RPC (n=100)": NEAR["stakes"],
        "Ципф α=0,5 (n=100)": zipf_stakes(100, 0.5),
        "Ципф α=1,0 (n=100)": zipf_stakes(100, 1.0),
        "Парето ξ=1,2 (n=100)": pareto_stakes(100, 1.2),
        "Парето ξ=2,0 (n=100)": pareto_stakes(100, 2.0),
        "Ципф α=1,0 (n=1000)": zipf_stakes(1000, 1.0),
        "Парето ξ=1,2 (n=1000)": pareto_stakes(1000, 1.2),
    }
    rows = []; curves = {}
    f = 1 / 3 - 1e-9
    for dname, st in dists.items():
        K = 400
        for strat in ["top", "bottom", "random"]:
            if strat == "random":
                ks = {e: [] for e in [40, 80]}; ksu = {e: [] for e in [40, 80]}; nadv = []
                for rep in range(15):
                    c = weighted_case(st, f, "random", K, rng=np.random.default_rng(SEED + rep))
                    for e in [40, 80]:
                        ks[e].append(kstar_from_curve(c["lp_w"], e)); ksu[e].append(kstar_from_curve(c["lp_u"], e))
                    nadv.append(c["n_adv"])
                rows.append(dict(distribution=dname, adversary="random (медіана 15)", n_adv=int(np.median(nadv)),
                                 **{f"W-noRepl k*(2^-{e})": float(np.nanmedian(ks[e])) for e in [40, 80]},
                                 **{f"Uniform k*(2^-{e})": float(np.nanmedian(ksu[e])) for e in [40, 80]},
                                 **{f"W-repl k*(2^-{e})": kstar_binom(f, e) for e in [40, 80]}))
                continue
            c = weighted_case(st, f, strat, K)
            curves[(dname, strat)] = c
            rows.append(dict(distribution=dname, adversary=strat, n_adv=c["n_adv"], n_min_quorum_S1prime=n_min_quorum(st),
                             A=round(c["A"], 4), H=round(c["H"], 4),
                             **{f"W-noRepl k*(2^-{e})": kstar_from_curve(c["lp_w"], e) for e in [40, 80]},
                             **{f"Uniform k*(2^-{e})": kstar_from_curve(c["lp_u"], e) for e in [40, 80]},
                             **{f"W-repl k*(2^-{e})": kstar_from_curve(c["lp_wr"], e) for e in [40, 80]}))
    T2 = pd.DataFrame(rows); T2.to_csv(os.path.join(RES, "T2_kstar_weighted.csv"), index=False)
    # проектне (найгірше для захисника) k* = максимум за стратегіями адверсарія
    agg = []
    for dname, st in dists.items():
        d = T2[T2["distribution"] == dname]
        agg.append(dict(distribution=dname, n_min_quorum_S1prime=n_min_quorum(st),
                        **{f"design W-noRepl k*(2^-{e})": d[f"W-noRepl k*(2^-{e})"].max() for e in [40, 80]},
                        **{f"design Uniform k*(2^-{e})": d[f"Uniform k*(2^-{e})"].max() for e in [40, 80]},
                        **{f"W-repl k*(2^-{e})": kstar_binom(1/3, e) for e in [40, 80]}))
    T2b = pd.DataFrame(agg); T2b.to_csv(os.path.join(RES, "T2b_design_kstar.csv"), index=False)
    out["T2b"] = T2b
    out["T2"] = T2; out["curves"] = curves; out["dists"] = dists

    # --- T3: валідація інтеграла Монте-Карло ---
    rows = []
    rng = np.random.default_rng(SEED)
    for dname in ["рівні (n=100)", "NEAR, знімок RPC (n=100)", "Ципф α=1,0 (n=100)", "Парето ξ=1,2 (n=100)"]:
        for strat in ["top", "bottom"]:
            c = curves[(dname, strat)]
            for k in [2, 4, 6, 8]:
                if k > c["n_adv"]:
                    continue
                exact = 2.0 ** c["lp_w"][k - 1]
                N = 400000
                mc = mc_weighted_noreplace(dists[dname], c["claimed"], c["adv"], k, N, rng)
                se = math.sqrt(max(mc * (1 - mc), 1e-300) / N)
                hyp = None
                if dname.startswith("рівні"):
                    hyp = 2.0 ** p_hyper(100, 1 / 3, k)
                rows.append(dict(distribution=dname, adversary=strat, k=k, P_integral=exact, P_MC=mc,
                                 MC_95CI_lo=max(0, mc - 1.96 * se), MC_95CI_hi=mc + 1.96 * se,
                                 P_hypergeom=hyp, z=(mc - exact) / se if se > 0 else np.nan))
    T3 = pd.DataFrame(rows); T3.to_csv(os.path.join(RES, "T3_validation_MC.csv"), index=False)
    out["T3"] = T3

    # --- T4: NEAR n=100 — вартість і стійкість стратегій на один акт ---
    n = 100
    st = NEAR["stakes"]
    cN = curves[("NEAR, знімок RPC (n=100)", "bottom")]   # найгірший для захисника випадок
    rows = []
    s1 = cost_S1(n); s1b = cost_S1(n, batch=True)
    s4 = cost_S4(n, ids=NEAR["next_ids"], present=NEAR["present"])
    for v in ["lo", "nom", "hi"]:
        pass
    s2n, s2lo, s2hi = cost_S2("nom"), cost_S2("lo"), cost_S2("hi")
    def add(name, c, bits, gas, note=""):
        rows.append(dict(strategy=name, signatures=c["sigs"], bytes=int(c["bytes"]),
                         Mcycles=round(c["cycles"] / 1e6, 2), security_bits=bits, evm_gas=gas, note=note))
    add("S1 повна перевірка кворуму", s1, LAMBDA_ED, f"{gas_S1(n, P['G_ed_EIP665']):.0f} (G_ed=2000)", "окремі перевірки")
    add("S1 (пакетна перевірка)", s1b, LAMBDA_ED, "", "батч 64, Bernstein et al.")
    add("S4 штатний легкий клієнт NEAR (Borsh, модель)", s4, LAMBDA_ED, "", f"JSON RPC виміряно: {NEAR['json_bytes']} Б")
    nq = n_min_quorum(st, NEAR["present"])
    s1p = dict(sigs=nq, bytes=s4["bytes"] - (int(NEAR["present"].sum()) - nq) * 65,
               cycles=nq * ed_cycles() + bytes_valset(n, ids=NEAR["next_ids"]) * P["sha256_cycles_per_byte"])
    add("S1′ мінімальний за стейком кворум (NEAR)", s1p, LAMBDA_ED, "", f"перевірка {nq} найбільших наявних підписів")
    add("S2 рекурсивний доказ (номінал)", s2n, LAMBDA_S2, f"{GAS_S2['Groth16 (A3)']:.0f} … {GAS_S2['UltraPLONK (Bhatt et al.)']}", "4,5 мс × 4,7 ГГц; 180 112 Б")
    add("S2 рекурсивний доказ (діапазон)", dict(sigs=0, bytes=s2lo["bytes"], cycles=s2lo["cycles"]), LAMBDA_S2, "", f"такти {s2lo['cycles']/1e6:.1f}–{s2hi['cycles']/1e6:.1f} млн; байти {s2lo['bytes']}–{s2hi['bytes']}")
    for e in [40, 64, 80, 100]:
        k_eq = kstar_equal(n, 1 / 3, e)
        k_near = kstar_from_curve(cN["lp_w"], e)
        c3 = cost_S3(n, k_near, merkle=True); c3f = cost_S3(n, k_near, merkle=False, ids=NEAR["next_ids"])
        add(f"S3 вибірка, ε=2^-{e}, NEAR-стейки (Меркл-зобов.)", c3, e, f"{gas_S3(n, k_near, P['G_ed_EIP665']):.0f} (G_ed=2000)", f"k*={k_near} (рівні стейки: {k_eq})")
        add(f"S3 вибірка, ε=2^-{e}, поточний NEAR (плоский хеш)", c3f, e, "", f"k*={k_near}")
    T4 = pd.DataFrame(rows); T4.to_csv(os.path.join(RES, "T4_costs_NEAR_n100.csv"), index=False)
    out["T4"] = T4

    # --- T5: точки перетину n* ---
    ns = np.unique(np.round(np.geomspace(10, 100000, 400)).astype(int))
    rows = []
    def first_cross(cost_a, cost_b):
        """найменше n, з якого cost_a(n) > cost_b(n) (a дорожча за b)."""
        for nn in ns:
            if cost_a(nn) > cost_b(nn):
                return int(nn)
        return None
    for v in ["lo", "nom", "hi"]:
        c2 = cost_S2(v)
        rows.append(dict(metric="CPU (цикли)", pair="S1 vs S2", variant=f"S2 {v}", n_star=first_cross(lambda nn: cost_S1(nn)["cycles"], lambda nn: c2["cycles"])))
        rows.append(dict(metric="CPU (цикли), S1 пакетна", pair="S1b vs S2", variant=f"S2 {v}", n_star=first_cross(lambda nn: cost_S1(nn, True)["cycles"], lambda nn: c2["cycles"])))
        rows.append(dict(metric="обсяг даних (Б)", pair="S1 vs S2", variant=f"S2 {v}", n_star=first_cross(lambda nn: cost_S1(nn)["bytes"], lambda nn: c2["bytes"])))
        for e in [40, 80]:
            for rname, r in [("r=0", 0.0), ("r=2^64", 2.0 ** 64)]:
                rows.append(dict(metric="CPU (цикли)", pair=f"S3(ε=2^-{e},{rname}) vs S2", variant=f"S2 {v}",
                                 n_star=first_cross(lambda nn: cost_S3(nn, kstar_equal(nn, 1/3, e, r))["cycles"], lambda nn: c2["cycles"])))
                rows.append(dict(metric="обсяг даних (Б)", pair=f"S3(ε=2^-{e},{rname}) vs S2", variant=f"S2 {v}",
                                 n_star=first_cross(lambda nn: cost_S3(nn, kstar_equal(nn, 1/3, e, r))["bytes"], lambda nn: c2["bytes"])))
    for gname, g2 in GAS_S2.items():
        for G_ed in [2000, 3000, 1e5, 1e6]:
            rows.append(dict(metric="газ EVM", pair="S1 vs S2", variant=f"S2 {gname}; G_ed={G_ed:g}", n_star=first_cross(lambda nn: gas_S1(nn, G_ed), lambda nn: g2)))
            rows.append(dict(metric="газ EVM", pair="S3(ε=2^-80,r=2^10) vs S2", variant=f"S2 {gname}; G_ed={G_ed:g}",
                             n_star=first_cross(lambda nn: gas_S3(nn, kstar_equal(nn, 1/3, 80, 2.0**10), G_ed), lambda nn: g2)))
    T5 = pd.DataFrame(rows); T5["n_star"] = T5["n_star"].astype("Int64")
    T5.to_csv(os.path.join(RES, "T5_crossovers.csv"), index=False)
    out["T5"] = T5

    # --- T6: граничний G_ed* (газ на одну перевірку Ed25519), за якого S3 = S2 ---
    rows = []
    for n_ in [100, 1000, 10000, 100000]:
        for e in [40, 80, 128]:
            k = kstar_equal(n_, 1 / 3, e, 2.0 ** 10)
            for gname, g2 in GAS_S2.items():
                lo, hi = 0.0, 1e7
                if gas_S3(n_, k, 0.0) >= g2:
                    Gs = 0.0
                else:
                    for _ in range(80):
                        mid = (lo + hi) / 2
                        (lo, hi) = (mid, hi) if gas_S3(n_, k, mid) < g2 else (lo, mid)
                    Gs = lo
                rows.append(dict(n=n_, eps=f"2^-{e}", k_star=k, S2_variant=gname, S2_gas=int(g2),
                                 S3_gas_at_G0=int(gas_S3(n_, k, 0.0)), G_ed_breakeven=int(Gs)))
    T6 = pd.DataFrame(rows); T6.to_csv(os.path.join(RES, "T6_Ged_breakeven.csv"), index=False)
    out["T6"] = T6

    # --- T7: деградація стійкості S3 при недооцінці f ---
    rows = []
    for n_ in [100, 1000, None]:
        for e in [40, 80]:
            k = kstar_binom(1/3, e) if n_ is None else kstar_equal(n_, 1/3, e)
            for f_true in [0.30, 1/3, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
                lp = p_binom_bound(f_true, k) if n_ is None else p_hyper(n_, f_true, k)
                rows.append(dict(n="∞" if n_ is None else n_, design_eps=f"2^-{e}", k=k, f_true=round(f_true, 3),
                                 bits_S3=-lp if np.isfinite(lp) else np.inf,
                                 bits_S1=LAMBDA_ED if f_true < 2/3 else 0.0, bits_S2=LAMBDA_S2 if f_true < 2/3 else 0.0))
    T7 = pd.DataFrame(rows); T7.to_csv(os.path.join(RES, "T7_degradation.csv"), index=False)
    out["T7"] = T7

    # --- NEAR-знімок: описові статистики ---
    srt = np.sort(st)[::-1]; cs = np.cumsum(srt)
    near_stats = dict(
        snapshot_height=NEAR["height"], epoch_id=NEAR["epoch_id"], n_bp=int(len(st)),
        approvals_present=int(NEAR["present"].sum()),
        signed_stake_fraction=float(st[NEAR["present"]].sum()),
        n_for_1_3=int(np.searchsorted(cs, 1/3) + 1), n_for_2_3=int(np.searchsorted(cs, 2/3, side="right") + 1),
        top1_share=float(srt[0]), top10_share=float(cs[9]), min_share=float(srt[-1]),
        gini=float(1 - 2 * np.sum(np.cumsum(np.sort(st)) ) / len(st) + 1 / len(st)),
        lcb_json_bytes=NEAR["json_bytes"], lcb_borsh_model_bytes=int(s4["bytes"]),
    )
    json.dump(near_stats, open(os.path.join(RES, "near_snapshot_stats.json"), "w"), ensure_ascii=False, indent=1)
    out["near_stats"] = near_stats
    return out

# ----------------------------------------------------------------------------
# 6. Теплові карти (n, f): оптимальна стратегія
# ----------------------------------------------------------------------------
def heatmaps(eps_bits=80):
    ns = np.unique(np.round(np.geomspace(10, 100000, 41)).astype(int))
    fs = np.round(np.arange(0.02, 0.70001, 0.02), 3)
    metrics = {
        "cpu": dict(r=0.0), "bytes": dict(r=0.0),
        "gas_2000": dict(r=2.0 ** 10, G=2000.0), "gas_1e5": dict(r=2.0 ** 10, G=1e5),
    }
    codes = {"S1": 0, "S2": 1, "S3": 2, "none": 3}
    res = {}
    rows = []
    for mname, mp in metrics.items():
        M = np.zeros((len(fs), len(ns)), int)
        for j, n in enumerate(ns):
            for i, f in enumerate(fs):
                if f * n >= quorum(n) or f >= 2/3:
                    M[i, j] = codes["none"]; continue
                k = kstar_equal(int(n), float(f), eps_bits, mp["r"])
                cand = {}
                if mname == "cpu":
                    cand["S1"] = cost_S1(n)["cycles"]; cand["S3"] = cost_S3(n, k)["cycles"]
                    if LAMBDA_S2 >= eps_bits: cand["S2"] = cost_S2("nom")["cycles"]
                elif mname == "bytes":
                    cand["S1"] = cost_S1(n)["bytes"]; cand["S3"] = cost_S3(n, k)["bytes"]
                    if LAMBDA_S2 >= eps_bits: cand["S2"] = cost_S2("nom")["bytes"]
                else:
                    cand["S1"] = gas_S1(n, mp["G"]); cand["S3"] = gas_S3(n, k, mp["G"])
                    if LAMBDA_S2 >= eps_bits: cand["S2"] = GAS_S2["Groth16 (A3)"]
                best = min(cand, key=cand.get)
                M[i, j] = codes[best]
                rows.append(dict(metric=mname, n=int(n), f=float(f), k_star=k, best=best, **{f"cost_{a}": b for a, b in cand.items()}))
        res[mname] = M
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"T8_heatmap_eps2^-{eps_bits}.csv"), index=False)
    return ns, fs, res

# ----------------------------------------------------------------------------
# 7. Рисунки
# ----------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

def style():
    g = globals()
    if "apply_figure_style" in g and callable(g["apply_figure_style"]):
        g["apply_figure_style"](font="Arial", sizes=(8, 7, 6))
    else:  # автономний запуск: еквівалентні параметри
        plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                             "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                             "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False,
                             "axes.spines.right": False, "savefig.dpi": 300, "legend.frameon": False,
                             "pdf.fonttype": 42})
    plt.rcParams["axes.unicode_minus"] = True

C = {"S1": "#4C72B0", "S2": "#C44E52", "S3": "#55A868", "S4": "#8172B2", "none": "#DDDDDD"}

def letter(ax, s):
    ax.text(-0.13, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")

def fig1_security_vs_k(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    ks = np.arange(1, 141)
    for n, ls in [(100, "-"), (1000, "--"), (10000, ":")]:
        y = [-p_hyper(n, 1/3, k) for k in ks]
        y = np.array([v if np.isfinite(v) else np.nan for v in y])
        ax.plot(ks, y, ls, color=C["S3"], lw=1.3, label=f"рівні стейки, n = {n}")
        kd = int(math.floor(n / 3)) + 1
        if kd <= ks[-1]:
            ax.axvline(kd, color=C["S3"], lw=0.6, ls=ls, alpha=0.5)
    ax.plot(ks, [-p_binom_bound(1/3, k) for k in ks], color="0.3", lw=1.0, label=r"межа з поверненням $2^{-k}$")
    for e in [40, 80, 128]:
        ax.axhline(e, color="0.75", lw=0.6)
        ax.text(2, e + 2, rf"$\varepsilon = 2^{{-{e}}}$", ha="left", va="bottom", fontsize=6, color="0.4")
    ax.set_xlabel("кількість перевірених підписів k")
    ax.set_ylabel(r"стійкість $-\log_2 P$(атаки), біт")
    ax.set_ylim(0, 200); ax.set_xlim(0, 141)
    ax.legend(loc="upper left", fontsize=6)
    letter(ax, "а")
    ax = axes[1]
    cur = out["curves"]
    for (dn, strat), col, ls, lab, ulab in [
        (("NEAR, знімок RPC (n=100)", "top"), C["S3"], "-", "NEAR, зловмисні — 7 найбільших і 26-й", None),
        (("NEAR, знімок RPC (n=100)", "bottom"), C["S3"], "--", "NEAR, зловмисні — 77 найменших", "NEAR, 77 найменших, рівноймовірна вибірка"),
        (("Парето ξ=1,2 (n=100)", "bottom"), C["S4"], "--", "Парето ξ = 1,2, найменші", "Парето ξ = 1,2, рівноймовірна вибірка")]:
        c = cur[(dn, strat)]
        y = np.array([-v if np.isfinite(v) else np.nan for v in c["lp_w"][:100]])
        ax.plot(np.arange(1, 101), y, ls, color=col, lw=1.3, label=lab)
        if ulab:
            yu = np.array([-v if np.isfinite(v) else np.nan for v in c["lp_u"][:100]])
            ax.plot(np.arange(1, 101), yu, ":", color=col, lw=1.0, label=ulab)
    ax.annotate("з k = 9 атака неможлива\n(усі 8 зловмисних уже вибрано)", xy=(8.3, 16), xytext=(12, 100), fontsize=6, color="0.3",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.5"))
    for e in [40, 80]:
        ax.axhline(e, color="0.75", lw=0.6)
    ax.set_xlabel("кількість перевірених підписів k")
    ax.set_ylim(0, 200); ax.set_xlim(0, 101)
    ax.legend(loc="upper left", fontsize=6)
    letter(ax, "б")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_security_vs_k.png"), dpi=300)
    plt.close(fig)

def fig2_kstar_and_degradation(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    fs = np.linspace(0.05, 0.65, 61)
    for (rname, r), ls in zip(R_SET.items(), ["-", "--", ":"]):
        y = [kstar_equal(1000, f, 80, r) for f in fs]
        ax.plot(fs, y, ls, color=C["S3"], lw=1.3, label=rname.replace("2^10", "$2^{10}$").replace("2^64", "$2^{64}$"))
    ax.plot(fs, [quorum(1000)] * len(fs), color=C["S1"], lw=1.0, label="кворум S1 (667)")
    ax.plot(fs, np.floor(fs * 1000) + 1, color="0.6", lw=0.7, ls="-.", label="детерміновано, k = a + 1")
    ax.set_yscale("log")
    ax.set_xlabel("частка зловмисного стейку f")
    ax.set_ylabel(r"мінімальне $k^*(\varepsilon = 2^{-80})$")
    ax.legend(loc="lower right", fontsize=6)
    letter(ax, "а")
    ax = axes[1]
    T7 = out["T7"]
    for n_, ls in [(100, "-"), (1000, "--"), ("∞", ":")]:
        d = T7[(T7["n"] == n_) & (T7["design_eps"] == "2^-80")]
        y = d["bits_S3"].replace(np.inf, np.nan)
        ax.plot(d["f_true"], y, ls, color=C["S3"], marker="o", ms=2.5, lw=1.2, label=f"S3, n = {n_}")
    ax.axhline(LAMBDA_ED, color=C["S1"], lw=1.0, label="S1/S4 (Ed25519)")
    ax.axhline(LAMBDA_S2, color=C["S2"], lw=1.0, label="S2 (Plonky2, оцінена стійкість)")
    ax.axhline(80, color="0.75", lw=0.6)
    ax.set_xlabel("фактична частка зловмисного стейку f")
    ax.set_ylabel("стійкість, біт")
    ax.set_ylim(0, 140)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.69), fontsize=6)
    letter(ax, "б")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_kstar_degradation.png"), dpi=300)
    plt.close(fig)

def fig3_pareto(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    cN = out["curves"][("NEAR, знімок RPC (n=100)", "bottom")]   # найгірший випадок
    n = 100
    nq = n_min_quorum(NEAR["stakes"], NEAR["present"])
    ks = np.arange(1, cN["n_adv"] + 2)
    bits = np.array([min(LAMBDA_ED, -cN["lp_w"][k - 1]) if np.isfinite(cN["lp_w"][k - 1]) else LAMBDA_ED for k in ks])
    for ax, metric, scale, lab in [(axes[0], "cycles", 1e6, "обчислення перевірки, млн тактів"),
                                    (axes[1], "bytes", 1e3, "обсяг даних, КБ")]:
        x3 = [cost_S3(n, k)[metric] / scale for k in ks]
        x3f = [cost_S3(n, k, merkle=False, ids=NEAR["next_ids"])[metric] / scale for k in ks]
        ax.plot(x3, bits, color=C["S3"], lw=1.4, marker="o", ms=1.8, label="S3 вибірка (зобов'язання на основі дерева Меркла)")
        if metric == "bytes":
            ax.plot(x3f, bits, color=C["S3"], lw=1.0, ls="--", label="S3 у поточному NEAR (повний next_bps)")
        s1 = cost_S1(n)[metric] / scale
        s4 = cost_S4(n, ids=NEAR["next_ids"], present=NEAR["present"])[metric] / scale
        ax.plot([s1], [LAMBDA_ED], "s", color=C["S1"], ms=5, label="S1 повна перевірка кворуму")
        ax.plot([s4], [LAMBDA_ED - 4], "D", color=C["S4"], ms=4.5, label="S4 штатний легкий клієнт NEAR")
        s4c = cost_S4(n, ids=NEAR["next_ids"], present=NEAR["present"])
        s1p = (nq * ed_cycles() + bytes_valset(n, ids=NEAR["next_ids"]) * P["sha256_cycles_per_byte"]) if metric == "cycles" else (s4c["bytes"] - (int(NEAR["present"].sum()) - nq) * 65)
        ax.plot([s1p / scale], [LAMBDA_ED], "^", color=C["S1"], ms=5, mfc="white", label=f"S1′ мінімальний кворум за стейком ({nq} підписів)")
        lo, nom, hi = cost_S2("lo")[metric] / scale, cost_S2("nom")[metric] / scale, cost_S2("hi")[metric] / scale
        ax.errorbar([nom], [LAMBDA_S2], xerr=[[nom - lo], [hi - nom]], fmt="o", color=C["S2"], ms=5, capsize=2, label="S2 доказ (діапазон вимірів)")
        if metric == "bytes":
            ax.plot([NEAR["json_bytes"] / scale], [LAMBDA_ED - 8], "D", mfc="none", color=C["S4"], ms=4.5, label="S4, JSON RPC (виміряно)")
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.margins(x=0.08)
    axes[0].set_ylabel(r"стійкість $-\log_2 P$, біт")
    axes[0].set_ylim(0, 140)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=5.5, bbox_to_anchor=(0.5, 0.0))
    letter(axes[0], "а"); letter(axes[1], "б")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(FIG, "fig3_pareto_near.png"), dpi=300)
    plt.close(fig)

def fig4_cost_vs_n(out):
    ns = np.unique(np.round(np.geomspace(10, 100000, 200)).astype(int))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    # CPU
    ax = axes[0]
    ax.plot(ns, [cost_S1(n)["cycles"] / 1e6 for n in ns], color=C["S1"], lw=1.3, label="S1")
    ax.plot(ns, [cost_S1(n, True)["cycles"] / 1e6 for n in ns], color=C["S1"], lw=0.9, ls="--", label="S1, пакетна")
    for e, ls in [(40, ":"), (80, "-")]:
        ax.plot(ns, [cost_S3(n, kstar_equal(n, 1/3, e))["cycles"] / 1e6 for n in ns], color=C["S3"], lw=1.2, ls=ls, label=rf"S3, $\varepsilon = 2^{{-{e}}}$")
    ax.fill_between(ns, cost_S2("lo")["cycles"] / 1e6, cost_S2("hi")["cycles"] / 1e6, color=C["S2"], alpha=0.25, lw=0)
    ax.axhline(cost_S2("nom")["cycles"] / 1e6, color=C["S2"], lw=1.3, label="S2 (смуга — діапазон)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("кількість валідаторів n"); ax.set_ylabel("млн тактів на крок")
    ax.set_title("Обчислення", loc="left"); ax.legend(fontsize=5.5, loc="upper left")
    letter(ax, "а")
    # bytes
    ax = axes[1]
    ax.plot(ns, [cost_S1(n)["bytes"] / 1e3 for n in ns], color=C["S1"], lw=1.3, label="S1")
    for e, ls in [(40, ":"), (80, "-")]:
        ax.plot(ns, [cost_S3(n, kstar_equal(n, 1/3, e))["bytes"] / 1e3 for n in ns], color=C["S3"], lw=1.2, ls=ls, label=rf"S3, $\varepsilon = 2^{{-{e}}}$")
    ax.plot(ns, [cost_S3(n, kstar_equal(n, 1/3, 80, 2.0**64))["bytes"] / 1e3 for n in ns], color=C["S3"], lw=0.9, ls="-.", label=r"S3, $\varepsilon = 2^{-80}$, $r = 2^{64}$")
    ax.fill_between(ns, cost_S2("lo")["bytes"] / 1e3, cost_S2("hi")["bytes"] / 1e3, color=C["S2"], alpha=0.25, lw=0)
    ax.axhline(cost_S2("nom")["bytes"] / 1e3, color=C["S2"], lw=1.3, label="S2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("кількість валідаторів n"); ax.set_ylabel("КБ на крок")
    ax.set_title("Обсяг даних", loc="left"); ax.legend(fontsize=5.5, loc="upper left")
    letter(ax, "б")
    # gas
    ax = axes[2]
    for G, ls in [(2000, "-"), (1e5, "--")]:
        ax.plot(ns, [gas_S1(n, G) / 1e6 for n in ns], color=C["S1"], lw=1.2, ls=ls, label=rf"S1, $G_{{ed}}$ = {G:g}")
        ax.plot(ns, [gas_S3(n, kstar_equal(n, 1/3, 80, 2.0**10), G) / 1e6 for n in ns], color=C["S3"], lw=1.2, ls=ls, label=rf"S3, $G_{{ed}}$ = {G:g}")
    ax.axhline(GAS_S2["Groth16 (A3)"] / 1e6, color=C["S2"], lw=1.3, label="S2, обгортка Groth16")
    ax.axhline(GAS_S2["UltraPLONK (Bhatt et al.)"] / 1e6, color=C["S2"], lw=1.0, ls="--", label="S2, рівень UltraPLONK")
    ax.axhline(gas_S2_direct_lower() / 1e6, color=C["S2"], lw=0.8, ls=":", label="S2 без обгортки (лише calldata)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("кількість валідаторів n"); ax.set_ylabel("млн газу на крок")
    ax.set_title(r"Газ EVM ($\varepsilon = 2^{-80}$, $r = 2^{10}$)", loc="left"); ax.legend(fontsize=5, loc="upper left")
    letter(ax, "в")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_cost_vs_n.png"), dpi=300)
    plt.close(fig)

def fig5_heatmaps(ns, fs, res, eps_bits=80):
    cmap = ListedColormap([C["S1"], C["S2"], C["S3"], C["none"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    titles = {"cpu": "Обчислення (r = 0)", "bytes": "Обсяг даних (r = 0)",
              "gas_2000": r"Газ, $G_{ed}$ = 2000 ($r = 2^{10}$)", "gas_1e5": r"Газ, $G_{ed} = 10^5$ ($r = 2^{10}$)"}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
    for ax, (m, lt) in zip(axes.flat, zip(titles, "абвг")):
        ax.pcolormesh(ns, fs, res[m], cmap=cmap, norm=norm, shading="nearest", rasterized=True)
        ax.set_xscale("log")
        ax.axhline(1/3, color="k", lw=0.6, ls="--")
        ax.set_title(titles[m], loc="left")
        letter(ax, lt)
    for ax in axes[1]:
        ax.set_xlabel("кількість валідаторів n")
    for ax in axes[:, 0]:
        ax.set_ylabel("частка зловмисного стейку f")
    handles = [Patch(color=C["S1"], label="S1 повна перевірка"), Patch(color=C["S2"], label="S2 рекурсивний доказ"),
               Patch(color=C["S3"], label="S3 вибірка k*"), Patch(color=C["none"], label="жодна (f ≥ 2/3)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(FIG, f"fig5_heatmaps_eps{eps_bits}.png"), dpi=300)
    plt.close(fig)

def fig6_weighted(out):
    T2 = out["T2"]
    d = T2[T2["adversary"].isin(["top", "bottom"])].copy()
    labels = [f"{r.distribution}\n{'найбільші' if r.adversary=='top' else 'найменші'}" for r in d.itertuples()]
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(d["W-noRepl k*(2^-80)"], y, color=C["S3"], s=18, zorder=3, label="зважена без повернення")
    ax.scatter(d["W-repl k*(2^-80)"], y, marker="|", color="0.2", s=60, zorder=3, label="зважена з поверненням")
    u = d["Uniform k*(2^-80)"]
    ax.scatter(u, y, marker="x", color="0.35", s=18, zorder=3, label="рівноймовірна вибірка (без стейку)")
    ax.scatter(d["n_min_quorum_S1prime"], y, marker="^", facecolor="white", edgecolor=C["S1"], s=20, zorder=4, label="S1′: мінімальний кворум (128 біт)")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"кількість перевірених підписів для $\varepsilon = 2^{-80}$ ($f = 1/3$, $r = 0$)")
    ax.legend(loc="upper right", fontsize=6)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_weighted_sampling.png"), dpi=300)
    plt.close(fig)

def main():
    t0 = time.time()
    style()
    out = run_tables()
    ns, fs, res = heatmaps(80)
    fig1_security_vs_k(out); fig2_kstar_and_degradation(out); fig3_pareto(out)
    fig4_cost_vs_n(out); fig5_heatmaps(ns, fs, res, 80); fig6_weighted(out)
    # версії середовища
    import scipy, matplotlib as mpl
    with open(os.path.join(HERE, "requirements.txt"), "w") as fh:
        fh.write(f"python=={sys.version.split()[0]}\nnumpy=={np.__version__}\nscipy=={scipy.__version__}\n"
                 f"pandas=={pd.__version__}\nmatplotlib=={mpl.__version__}\n")
    print(f"done in {time.time()-t0:.1f} s")
    return out

if __name__ == "__main__":
    main()
