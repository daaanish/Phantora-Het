"""Stream a debug-level phantora_server log into a compact per-event table."""
import gzip, json, re, sys
src, dst = sys.argv[1:3]
started = re.compile(r"(\d+): Event (\S+) started, (.*)$")
ended = re.compile(r"(\d+): Event (\S+) ended \(started at (\d+)\)")
comp = re.compile(r"Computation\((.*?), (-?\d+), (Torch|Cuda)\((.*)$")
info = re.compile(r"info: (\w+)")
qop = re.compile(r'op: "?(\w+)"?')
cuda = re.compile(r"^(\w+)")
sigs, events, n = {}, {}, 0
with open(src, errors="replace") as f, gzip.open(dst, "wt") as out:
    for line in f:
        if "Event " not in line:
            continue
        m = started.search(line)
        if m:
            t, eid, action = int(m.group(1)), m.group(2), m.group(3)
            c = comp.match(action)
            if c:
                node, dur, kind, rest = c.group(1), int(c.group(2)), c.group(3), c.group(4)
                if kind == "Torch":
                    i = info.search(rest)
                    label = i.group(1) if i else "?"
                    body = rest[i.start():] if i else rest
                    if label == "QwenOp":
                        q = qop.search(body)
                        label = "QwenOp:" + (q.group(1) if q else "?")
                else:
                    label = "cuda:" + cuda.match(rest).group(1)
                    body = rest
                kindc = "T" if kind == "Torch" else "C"
            else:
                node, dur, kindc, label, body = "", -1, "N", "comm", action
                b = re.search(r"bytes: (\d+)", action)
                s = re.search(r'src: "([^"]*)"', action)
                node = s.group(1) if s else ""
                dur = int(b.group(1)) if b else -1   # bytes for comm rows
                c2 = re.search(r"\}, (\w+)", action)
                label = "comm:" + (c2.group(1) if c2 else "?")
            body = body[:400]
            sid = sigs.setdefault(body, len(sigs))
            events[eid] = (t, node, kindc, label, dur, sid)
            continue
        m = ended.search(line)
        if m and m.group(2) in events:
            t, node, kindc, label, dur, sid = events.pop(m.group(2))
            out.write(json.dumps([t, int(m.group(1)), node, kindc, label, dur, sid]) + "\n")
            n += 1
    out.write(json.dumps({"sigs": {v: k for k, v in sigs.items()}, "unended": len(events), "events": n}) + "\n")
print("events", n, "unended", len(events), "sigs", len(sigs))
