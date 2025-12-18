import os
import hashlib
import json
import math
import tkinter as tk
from tkinter import scrolledtext, messagebox
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from typing import List, Tuple

# Argon2
try:
    from argon2.low_level import hash_secret_raw, Type
except ImportError:
    print("UYARI: Argon2 kütüphanesi (argon2-cffi) yüklü değil. Sadece Möbiüs Fold çalışacaktır.")
    hash_secret_raw = None
    Type = None

# ---------- Precision & constants ----------
getcontext().prec = 80
SCALE = 10 ** 9
DEC_PI = Decimal(str(math.pi))
DEC_TWO = Decimal(2)
DEC_TWO_PI = DEC_PI * DEC_TWO

# ---------- Decimal trig helpers ----------
def _reduce_angle(x: Decimal) -> Decimal:
    two_pi = DEC_TWO_PI
    n = (x / two_pi).to_integral_value(rounding=ROUND_HALF_EVEN)
    x = x - n * two_pi
    if x > DEC_PI:
        x -= two_pi
    if x < -DEC_PI:
        x += two_pi
    return x

def dec_sin(x: Decimal) -> Decimal:
    x = _reduce_angle(x)
    term = x
    s = term
    x2 = x * x
    sign = -1
    for n in range(1, 20):
        denom = Decimal((2 * n) * (2 * n + 1))
        term = term * x2 / denom * sign
        s += term
        sign *= -1
    return +s

def dec_cos(x: Decimal) -> Decimal:
    x = _reduce_angle(x)
    term = Decimal(1)
    s = term
    x2 = x * x
    sign = -1
    for n in range(1, 20):
        denom = Decimal((2 * n - 1) * (2 * n))
        term = term * x2 / denom * sign
        s += term
        sign *= -1
    return +s

# ---------- Möbiüs Fonksiyonları ----------
def generate_mobius_integer_points_seeded(seed_bytes: bytes, num_points: int = 512, num_twists: int = 2, orientation: int = 1):
    points = []
    prev_hash = seed_bytes
    SCALE_DEC = Decimal(SCALE)

    for i in range(num_points):
        h = hashlib.sha256(prev_hash).digest()
        u_int = int.from_bytes(h[:8], "big")
        v_int = int.from_bytes(h[8:16], "big")
        u_dec = Decimal(u_int % 10**9) / Decimal(10**9)
        v_dec = Decimal(v_int % 10**9) / Decimal(10**9)
        u_full = u_dec * DEC_TWO_PI
        v_half_width = Decimal("0.1")
        v_i_decimal = (v_dec * DEC_TWO - Decimal(1)) * v_half_width

        # Twist factor topolojik parametreleri yansıtıyor
        twist_factor = Decimal(num_twists) * u_full / DEC_TWO * (1 if orientation else -1)

        cos_half = dec_cos(twist_factor)
        sin_half = dec_sin(twist_factor)
        cos_u = dec_cos(u_full)
        sin_u = dec_sin(u_full)

        one = Decimal(1)
        x_dec = (one + v_i_decimal * cos_half) * cos_u
        y_dec = (one + v_i_decimal * cos_half) * sin_u
        z_dec = v_i_decimal * sin_half

        X_i = int((x_dec * SCALE_DEC).to_integral_value(rounding=ROUND_HALF_EVEN))
        Y_i = int((y_dec * SCALE_DEC).to_integral_value(rounding=ROUND_HALF_EVEN))
        Z_i = int((z_dec * SCALE_DEC).to_integral_value(rounding=ROUND_HALF_EVEN))

        points.append(X_i.to_bytes(8, 'big', signed=True) +
                      Y_i.to_bytes(8, 'big', signed=True) +
                      Z_i.to_bytes(8, 'big', signed=True))

        coord_bytes = X_i.to_bytes(16, "big", signed=True) + \
                      Y_i.to_bytes(16, "big", signed=True) + \
                      Z_i.to_bytes(16, "big", signed=True)
        prev_hash = hashlib.sha256(h + coord_bytes).digest()

    return b"".join(points), prev_hash

def mobius_fold(k_material: bytes, fold_iters: int = 100, num_points: int = 512, num_twists: int = 2, orientation: int = 1):
    prev_hash = k_material
    log = {}
    for i in range(fold_iters):
        mobius_bytes, feedback_hash = generate_mobius_integer_points_seeded(prev_hash, num_points=num_points, num_twists=num_twists, orientation=orientation)
        prev_hash = feedback_hash
        if i % max(1, fold_iters // 10) == 0:
            log[f"iter_{i}"] = prev_hash.hex()
    final_mobius = prev_hash
    return final_mobius, log
def generate_deterministic_salt(password: str, secret: str,
                                num_twists: int, orientation: int,
                                fold_iters: int, num_points: int,
                                salt_len: int = 32) -> bytes:
    """
    Deterministik salt üretir: Möbiüs parametreleri + password + secret.
    """
    h = hashlib.sha256()

    h.update(password.encode("utf-8"))
    h.update(b"||")
    h.update(secret.encode("utf-8"))
    h.update(b"||")

    # Möbiüs topoloji parametreleri
    h.update(num_twists.to_bytes(2, "big"))
    h.update(orientation.to_bytes(1, "big"))
    h.update(fold_iters.to_bytes(4, "big"))
    h.update(num_points.to_bytes(4, "big"))

    out = h.digest()

    # istenen salt uzunluğuna kes
    return out[:salt_len]

def generate_salt(length: int = 32) -> bytes:
    return os.urandom(length)

def mobius_kdf(password: str, secret: str,
               fold_iters: int = 100, num_points: int = 512, num_twists: int = 2, orientation: int = 1,
               argon_time: int = 4, argon_memory_mb: int = 64, argon_parallel: int = 2,
               salt_len: int = 32):
    salt = generate_deterministic_salt(password, secret,
                                   num_twists, orientation,
                                   fold_iters, num_points,
                                   salt_len)
    # Seed artık topoloji ile birlikte
    topo_bits = num_twists.to_bytes(2, 'big') + orientation.to_bytes(1, 'big')
    initial = hashlib.sha256(password.encode("utf-8") + b"||" +
                             secret.encode("utf-8") + b"||" +
                             salt + topo_bits).digest()

    mobius_final, log = mobius_fold(initial, fold_iters, num_points, num_twists, orientation)
    combined_salt = mobius_final + salt
    mem_kb = max(8, argon_memory_mb * 1024)

    if hash_secret_raw:
        final_key = hash_secret_raw(
            secret=password.encode("utf-8"),
            salt=combined_salt,
            time_cost=argon_time,
            memory_cost=mem_kb,
            parallelism=argon_parallel,
            hash_len=32,
            type=Type.ID
        )
    else:
        final_key = hashlib.sha256(mobius_final + password.encode("utf-8")).digest()

    return {
        "final_key": final_key,
        "salt": salt,
        "mobius_final": mobius_final,
        "log": log
    }

# ---------- TKINTER ARAYÜZ ----------
def run_kdf():
    password = entry_password.get()
    secret = entry_secret.get()

    if not password or not secret:
        messagebox.showerror("Hata", "Lütfen tüm alanları doldurun.")
        return

    result = mobius_kdf(password, secret)

    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"--- SONUÇ ---\n", "title")
    output_text.insert(tk.END, f"Final Key (hex): {result['final_key'].hex()}\n")
    output_text.insert(tk.END, f"Salt (hex): {result['salt'].hex()}\n")
    output_text.insert(tk.END, f"Mobius Final (hex): {result['mobius_final'].hex()}\n\n")
    output_text.insert(tk.END, f"--- İlk 5 İterasyon ---\n", "title")
    for k, v in list(result["log"].items())[:5]:
        output_text.insert(tk.END, f"{k}: {v[:12]}...\n")

# ---------- UI TASARIMI ----------
root = tk.Tk()
root.title("Möbiüs Fold Key Derivation Function")
root.geometry("680x500")
root.config(bg="white")

title_label = tk.Label(root, text="🔴 Möbiüs Fold KDF", bg="white", fg="red", font=("Helvetica", 20, "bold"))
title_label.pack(pady=10)

frame_input = tk.Frame(root, bg="white")
frame_input.pack(pady=10)

tk.Label(frame_input, text="Parola:", bg="white", fg="black", font=("Arial", 12)).grid(row=0, column=0, sticky="e", padx=5)
entry_password = tk.Entry(frame_input, show="*", width=40, font=("Consolas", 11))
entry_password.grid(row=0, column=1, padx=5, pady=5)

tk.Label(frame_input, text="Gizli Modül:", bg="white", fg="black", font=("Arial", 12)).grid(row=1, column=0, sticky="e", padx=5)
entry_secret = tk.Entry(frame_input, show="*", width=40, font=("Consolas", 11))
entry_secret.grid(row=1, column=1, padx=5, pady=5)

btn_run = tk.Button(root, text="Anahtar Üret", command=run_kdf,
                    bg="red", fg="white", font=("Arial", 12, "bold"), relief="flat", padx=15, pady=6)
btn_run.pack(pady=10)

output_text = scrolledtext.ScrolledText(root, width=75, height=15, bg="white", fg="black", font=("Consolas", 10))
output_text.tag_config("title", foreground="red", font=("Arial", 11, "bold"))
output_text.pack(padx=10, pady=10)

root.mainloop()
