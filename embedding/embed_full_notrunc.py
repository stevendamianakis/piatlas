#!/usr/bin/env python3
"""
No-truncation embedding: embed the FULL text of every template (max_seq_length large
enough that nothing is cut). Uses length-sorted, token-budget adaptive batching so long
sequences get tiny batches (memory-safe) and short ones get big batches (no padding
waste). One model can serve several configs (dir::instruction) in a single process to
share the weights. Checkpointed per super-chunk for resume.
"""
import os, json, time, argparse, glob
import numpy as np


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def encode_batch(model, texts, prompt, kw):
    """Encode one batch; on CUDA OOM, split in half and retry recursively."""
    import torch
    try:
        return model.encode(texts, batch_size=len(texts), prompt=prompt, **kw)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() and len(texts) > 1:
            torch.cuda.empty_cache()
            mid = len(texts) // 2
            a = encode_batch(model, texts[:mid], prompt, kw)
            b = encode_batch(model, texts[mid:], prompt, kw)
            return np.concatenate([a, b], 0)
        raise


def make_batches(order, lengths, budget):
    """Greedily group sorted indices so count*max_len <= budget (>=1 each)."""
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * max(lengths[j] for j in cur + [i]) > budget:
            batches.append(cur); cur = [i]
        else:
            cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def embed_config(model, texts, lengths, order, instruction, out_dir, budget, super_chunk):
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, "emb_full.npy")
    if os.path.isfile(final):
        log(f"  {final} exists; skip config."); return
    ck = os.path.join(out_dir, "ckpt_nt"); os.makedirs(ck, exist_ok=True)
    kw = dict(normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    prompt = f"Instruct: {instruction}\nQuery: " if instruction.strip() else None
    N = len(texts); t0 = time.time(); done = 0
    # process sorted order in super-chunks (checkpoint unit)
    sc_bounds = list(range(0, N, super_chunk))
    for si, s in enumerate(sc_bounds):
        ckf = os.path.join(ck, f"sc_{si:05d}.npz")
        chunk_idx = order[s:s + super_chunk]
        if os.path.isfile(ckf):
            done += len(chunk_idx); continue
        embs, idxs = [], []
        for batch in make_batches(chunk_idx, lengths, budget):
            e = encode_batch(model, [texts[j] for j in batch], prompt, kw)
            embs.append(np.asarray(e, dtype=np.float16)); idxs.extend(batch)
        np.savez(ckf, emb=np.concatenate(embs, 0), idx=np.asarray(idxs))
        done += len(chunk_idx)
        rate = done / max(1e-9, time.time() - t0)
        log(f"  [{out_dir}] super-chunk {si+1}/{len(sc_bounds)}  {done}/{N}  "
            f"{rate:.1f} texts/s  ETA {(N-done)/max(1e-9,rate)/60:.1f} min "
            f"(maxlen in chunk={max(lengths[j] for j in chunk_idx)})")
    # assemble in original order
    dim = None
    parts = sorted(glob.glob(os.path.join(ck, "sc_*.npz")))
    emb_all = None
    for p in parts:
        d = np.load(p); e, ix = d["emb"], d["idx"]
        if emb_all is None:
            dim = e.shape[1]; emb_all = np.zeros((N, dim), dtype=np.float16)
        emb_all[ix] = e
    np.save(final, emb_all)
    log(f"  saved {final} {emb_all.shape} in {(time.time()-t0)/60:.1f} min")
    for p in parts:
        os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", action="append", required=True,
                    help="dir::instruction  (instruction empty = plain). Repeatable.")
    ap.add_argument("--texts", default="merged_texts.json")
    ap.add_argument("--maxlen", type=int, default=16384)
    ap.add_argument("--budget", type=int, default=12288)
    ap.add_argument("--super-chunk", type=int, default=2000)
    a = ap.parse_args()

    texts = [str(t) for t in json.load(open(a.texts))]
    N = len(texts)
    from transformers import AutoTokenizer
    log(f"tokenizing {N} texts for length sort ...")
    tok = AutoTokenizer.from_pretrained(a.model)
    lengths = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
    order = np.argsort(lengths)  # ascending: short batches first
    log(f"lengths: median={int(np.median(lengths))} p99={int(np.percentile(lengths,99))} "
        f"max={int(lengths.max())}; maxlen={a.maxlen} budget={a.budget}")

    import torch
    from sentence_transformers import SentenceTransformer
    log(f"loading {a.model} on cuda:0 (bfloat16) ...")
    model = SentenceTransformer(a.model, device="cuda:0", model_kwargs={"dtype": torch.bfloat16})
    model.max_seq_length = a.maxlen

    for spec in a.config:
        out_dir, _, instruction = spec.partition("::")
        log(f"=== config: dir={out_dir} instruction={'<plain>' if not instruction.strip() else instruction[:50]+'...'} ===")
        embed_config(model, texts, lengths, order, instruction, out_dir, a.budget, a.super_chunk)
    log("ALL CONFIGS DONE")


if __name__ == "__main__":
    main()
