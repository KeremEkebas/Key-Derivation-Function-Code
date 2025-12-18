import os, math, hashlib, numpy as np, scipy.stats as stats

try:
    from argon2.low_level import hash_secret_raw, Type
    HAS_ARGON2 = True
except Exception:
    HAS_ARGON2 = False

from mobius_kdf_module import mobius_kdf  


def normalize_key_maybe(result):
    """
    result: bytes OR str(hex) OR dict containing "final_key"
    returns: bytes
    """
    if isinstance(result, dict):
        if "final_key" in result:
            key = result["final_key"]
        else:
           
            key = str(result)
    else:
        key = result

    if isinstance(key, (bytes, bytearray)):
        return bytes(key)

    if isinstance(key, int):
        length = (key.bit_length() + 7) // 8 or 1
        return key.to_bytes(length, "big")

    if isinstance(key, str):
        s = key.strip()
        
        try:
            if all(c in "0123456789abcdefABCDEF" for c in s) and len(s) % 2 == 0:
                return bytes.fromhex(s)
        except Exception:
            pass
        
        return s.encode("utf-8")

    return str(key).encode("utf-8")


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0

    counts = {}
    for b in data:
        counts[b] = counts.get(b, 0) + 1

    total = len(data)
    ent = 0.0
    for v in counts.values():
        p = v / total
        ent -= p * math.log2(p)

    return ent


def hamming_distance(a, b) -> float:
    """
    a,b: bytes-like or convertible to bytes
    returns fraction of differing bits (0..1)
    """
    A = normalize_key_maybe(a)
    B = normalize_key_maybe(b)

    
    n = max(len(A), len(B))
    if len(A) < n:
        A = A + b"\x00" * (n - len(A))
    if len(B) < n:
        B = B + b"\x00" * (n - len(B))

    diff_bits = 0
    for x, y in zip(A, B):
        diff_bits += bin(x ^ y).count("1")

    return diff_bits / (n * 8)


def bit_independence(data: bytes) -> float:
    """
    Simple proxy: unpack bits, reshape to (n_bytes, 8) and compute correlation
    matrix, then return mean absolute deviation from identity.
    For single key this is a proxy; ideally compute over many keys.
    """
    b = normalize_key_maybe(data)
    if len(b) == 0:
        return 0.0

    arr = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    try:
        arr_reshaped = arr.reshape(len(b), 8)
    except Exception:
        
        arr_reshaped = arr.reshape(1, -1)

    corr = np.corrcoef(arr_reshaped, rowvar=False)
    if corr.size == 1:
        return 0.0

    identity = np.eye(corr.shape[0])
    return float(np.mean(np.abs(corr - identity)))


def avalanche_effect(func_callable, password: bytes, secret: bytes) -> float:
    pw = password if isinstance(password, bytes) else password.encode()
    sec = secret if isinstance(secret, bytes) else secret.encode()

    res_orig = func_callable(pw.decode("utf-8"), sec.decode("utf-8"))
    orig_key = normalize_key_maybe(res_orig)

    flipped = bytearray(pw)
    flipped[0] = flipped[0] ^ 1

    res_flip = func_callable(
        bytes(flipped).decode("utf-8"),
        sec.decode("utf-8")
    )
    flip_key = normalize_key_maybe(res_flip)

    return hamming_distance(orig_key, flip_key)


def wrapper_mobius(pw: str, sec: str):
    return mobius_kdf(pw, sec)  


def wrapper_argon2(pw: str, sec: str):
    if not HAS_ARGON2:
        
        s = hashlib.sha256(pw.encode() + b"||" + sec.encode()).digest()
        return {"final_key": s}

    salt = os.urandom(32)
    k = hash_secret_raw(
        secret=pw.encode(),
        salt=salt,
        time_cost=4,
        memory_cost=64 * 1024,
        parallelism=2,
        hash_len=32,
        type=Type.ID
    )
    return {"final_key": k}


def wrapper_sha512(pw: str, sec: str):
    salt = os.urandom(32)
    k = hashlib.pbkdf2_hmac("sha512", pw.encode(), salt, 100000, 32)
    return {"final_key": k}


def run_single(func_callable, password: str, secret: str):
    res = func_callable(password, secret)
    key = normalize_key_maybe(res)

    return {
        "final_key": key,
        "entropy": shannon_entropy(key),
        "avalanche": avalanche_effect(
            func_callable,
            password.encode(),
            secret.encode()
        ),
        "hamming": hamming_distance(key, os.urandom(len(key))),
        "bit_ind": bit_independence(key)
    }


def run_statistical_tests(n_trials=100):
    password = "mobius_cryptography"
    secret = "topology_key"

    methods = {
        "Mobius": wrapper_mobius,
        "Argon2": wrapper_argon2,
        "SHA512": wrapper_sha512
    }

    results = {
        name: {"entropy": [], "avalanche": [], "hamming": [], "bit_ind": []}
        for name in methods
    }

    for i in range(n_trials):
        for name, func in methods.items():
            r = run_single(func, password, secret)
            for m in ["entropy", "avalanche", "hamming", "bit_ind"]:
                results[name][m].append(r[m])

    summary = {}

    for name, metrics in results.items():
        print(f"\n--- {name} ---")
        summary[name] = {}

        for metric, vals in metrics.items():
            vals = np.array(vals, dtype=float)
            mean = vals.mean()
            std = vals.std(ddof=1)

            try:
                _, p_shapiro = stats.shapiro(vals) if len(vals) >= 3 else (None, 1.0)
            except Exception:
                p_shapiro = 1.0

            summary[name][metric] = {
                "mean": mean,
                "std": std,
                "shapiro_p": p_shapiro
            }

            print(
                f"{metric:12}: mean={mean:.4f} "
                f"std={std:.4f} shapiro_p={p_shapiro:.4f}"
            )

    print("\n--- pairwise t-tests (two-sided) on 'avalanche' metric ---")
    a = np.array(results["Mobius"]["avalanche"])
    b = np.array(results["Argon2"]["avalanche"])
    c = np.array(results["SHA512"]["avalanche"])

    t_ab = stats.ttest_ind(a, b, equal_var=False)
    t_ac = stats.ttest_ind(a, c, equal_var=False)

    print(f"Mobius vs Argon2: t={t_ab.statistic:.4f} p={t_ab.pvalue:.4f}")
    print(f"Mobius vs SHA512 : t={t_ac.statistic:.4f} p={t_ac.pvalue:.4f}")

    return results, summary


if __name__ == "__main__":
    print("Running statistical tests (this may take a minute)...")
    results, summary = run_statistical_tests(100)
    print("Done. You can inspect 'results' dict for per-trial values.")
